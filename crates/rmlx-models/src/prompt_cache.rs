// LOC-exempt: the PromptCacheEntry + SpillSink trait contracts, their blanket
// impl over SsdSpiller, and the PromptCache<E> container form one cohesion unit;
// splitting the blanket impl from the trait it serves would scatter the
// prompt-cache eviction/spill policy across modules.
//! Arch-agnostic prompt-cache primitives.
//!
//! ## Design
//!
//! `PromptCache<E>` is a multi-slot cache generic over an entry type `E`
//! that implements `PromptCacheEntry`. Each arch supplies its own concrete
//! entry type holding whatever post-prefill state it needs:
//!
//! - `Gemma4Entry` — `Vec<KvCache>` only (pure-attention arch).
//! - `Qwen35MoeEntry` — `Vec<KvCache>` + `Vec<LinearAttnCache>` (hybrid GDN).
//!
//! The cache is intentionally NOT a boxed-trait object or a single global.
//! Each arch owns its own `Mutex<Option<PromptCache<E>>>` static — identical
//! to the old Qwen-only pattern, just refactored so both archs share the
//! lookup / push / stats logic.
//!
//! ## Trait contract
//!
//! `PromptCacheEntry` exposes:
//! - `prompt_token_ids() -> &[u32]` — key for prefix matching.
//! - `deep_clone() -> Result<Self>` — cheap refcount clone of MLX arrays.
//! - `truncate_kv_to(prefix_len: usize)` — trim KV caches to prefix length.
//! - `kv_bytes() -> u64` — RAM estimate for eviction budget.
//!
//! ## Stats
//!
//! `CacheStats` is a simple flat struct: `hits`, `misses`, `evictions`, `bytes`.
//! It is updated on every `find_best_prefix` call (hit/miss) and every
//! evicting `push`. Reset via `clear()`.
//!
//! ## LRU policy (M29)
//!
//! Each slot carries a `last_used_seq: u64` counter (monotonically increasing,
//! no syscall). On every `find_best_prefix` hit the winning slot's counter is
//! bumped to the current global sequence. On `push`:
//!
//!   1. RAM cap check: if `total_kv_bytes + new_entry_bytes > max_bytes`,
//!      repeatedly evict the slot with the smallest `last_used_seq` (LRU)
//!      until the budget fits or slots are empty.
//!   2. Slot count cap: if `slots.len() == capacity`, evict the slot with the
//!      smallest `last_used_seq`.
//!
//! The smaller of the two caps (RAM bytes, slot count) wins — either can
//! trigger eviction first. Eviction never happens inside `find_best_prefix`
//! (finish lookup first, then evict).
//!
//! RAM cap is configurable via `--prompt-cache-ram-gb` CLI flag. Default: 2 GiB.

#![allow(clippy::struct_field_names)]
use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;

use crate::kv_cache::kv_layer_quants;
use crate::prefix_index::{
    active_prefix_index_kind, build_active_index, PrefixIndex, PrefixIndexKind,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{SpillJob, SsdHydrator, SsdSpiller};

// The SSD-tier hashing primitives (`FNV_OFFSET`, `FNV_PRIME`,
// `BLOCK_TOKENS`, `chained_block_hashes`, `chained_block_hashes_seeded`) and
// the `SsdHydrate<E>` trait live in `rmlx-kv-ssd` so the SSD modules can use
// them without a back-edge into `rmlx-models`. The re-exports below preserve
// every in-crate `crate::prompt_cache::FNV_OFFSET` / `BLOCK_TOKENS` /
// `chained_block_hashes_seeded` / `SsdHydrate` import path byte-identically.
pub(crate) use rmlx_kv_ssd::{
    chained_block_hashes_seeded, SsdHydrate, BLOCK_TOKENS, FNV_OFFSET, FNV_PRIME,
};
// `chained_block_hashes` (the un-seeded form) is no longer used by this module
// directly — `find_best_prefix` now always salts with a caller-supplied seed.
// It is still consumed by sibling `#[path]` test modules and by
// `gemma4::prompt_cache`, which import it directly from `rmlx_kv_ssd`.

/// Stable per-model identity, folded into the prompt-cache key by
/// [`cache_seed`].
///
/// Derived from the snapshot directory's own name (its registry id, e.g.
/// `mlx-community__gemma-4-e2b-it-mxfp8`) rather than the full path, so the
/// same model resolves to the same signature whichever absolute path it was
/// loaded through — the SSD tier's `.kvb` digests are seeded with this, and
/// they have to survive a restart.
pub(crate) fn model_cache_sig(model_dir: &std::path::Path) -> u64 {
    let id = model_dir
        .file_name()
        .map_or_else(|| model_dir.as_os_str(), |name| name)
        .to_string_lossy();
    crate::multimodal_cache::model_sig(&id)
}

// The seed formula itself is defined in `rmlx-kv-ssd` next to the FNV walk it
// feeds, because the SSD hydrate probe has to produce the same digest stream as
// this RAM cache and cannot see `rmlx-models`. Do NOT reintroduce a copy of the
// formula here — see `rmlx_kv_ssd::hashing::cache_seed`.
use rmlx_kv_ssd::cache_seed;

/// The prompt-cache / SSD digest seed for **one request**.
///
/// Wraps [`rmlx_kv_ssd::cache_seed`] with the one argument that crate cannot
/// produce: the per-layer codec vector `kv_quant` resolves to under this
/// build's layer policy. That policy lives in [`crate::kv_cache`], above the
/// SSD crate, so the vector is supplied from here — from the same
/// [`kv_layer_quants`] every arch builds its caches from.
///
/// Take `n_layers` rather than the vector so there is exactly one place in this
/// crate that expands it. A caller that built the vector for its caches and a
/// caller that only knows `n_layers` then cannot seed differently, which is the
/// failure that matters here: a push seeded differently from the query does not
/// return a wrong answer, it returns a cache that silently never hits.
///
/// Why per request, when `layout_key` already folds a vector: the layout key is
/// fixed at attach from the launch codec, and a request may resolve a different
/// one (`kv_quant_for_ctx` in auto mode, or an explicit per-request override).
/// For those requests the attach-time vector describes a layout they are not
/// running, so the guarantee "a layer-policy change invalidates stored blocks"
/// would hold only for requests on the launch default. Seeding from the
/// request's own mixture makes it hold for every request.
pub(crate) fn request_cache_seed(
    layout_key: u64,
    kv_quant: KvQuant,
    n_layers: usize,
    model_sig: u64,
) -> u64 {
    cache_seed(
        layout_key,
        kv_quant,
        &kv_layer_quants(n_layers, kv_quant),
        model_sig,
    )
}

/// Default RAM cap for the prompt cache (2 GiB).
pub(crate) const DEFAULT_MAX_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Resolve the RAM cap for the prompt cache from the CLI flag.
///
/// Precedence: CLI `--prompt-cache-ram-gb` > default 2 GiB.
///
/// `cli_gib` is the parsed `--prompt-cache-ram-gb` value (`Option<f64>` in GiB).
/// Negative / non-finite values fall through to the default.
/// CLI-supplied gibibytes are converted to bytes here.
///
/// Process-global resolution: stored in a `OnceLock` so every per-arch
/// `PromptCache::new` call sees the same value regardless of construction
/// order. The first call to [`install_ram_cap`] wins; later calls are
/// no-ops with a `warn!` if they disagree, matching the SSD-tier OnceLock.
pub fn resolve_ram_cap_bytes(cli_gib: Option<f64>) -> u64 {
    if let Some(g) = cli_gib {
        if g.is_finite() && g >= 0.0 {
            return (g * 1024.0 * 1024.0 * 1024.0) as u64;
        }
    }
    DEFAULT_MAX_BYTES
}

static RAM_CAP_BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Install the process-global RAM cap for [`PromptCache::new`].
///
/// First call wins. Pass `cli_gib = None` to use the 2 GiB default. See
/// [`resolve_ram_cap_bytes`] for precedence rules. Called once at serve
/// startup before any model loads. Idempotent re-entry: a second call with a
/// different value is dropped with a `warn!` (mirrors `ssd_tier::install_config`).
pub fn install_ram_cap(cli_gib: Option<f64>) {
    let bytes = resolve_ram_cap_bytes(cli_gib);
    if RAM_CAP_BYTES.set(bytes).is_err() {
        let existing = RAM_CAP_BYTES.get().copied().unwrap_or(0);
        if existing != bytes {
            tracing::warn!(
                existing,
                requested = bytes,
                "install_ram_cap called more than once; keeping the first value"
            );
        }
        return;
    }
    let source = if cli_gib.is_some_and(|g| g.is_finite() && g >= 0.0) {
        "cli"
    } else {
        "default"
    };
    tracing::info!(
        bytes,
        gib = (bytes as f64) / (1024.0 * 1024.0 * 1024.0),
        source,
        "prompt-cache RAM cap installed"
    );
}

/// Active RAM cap. Reads the [`install_ram_cap`] value; if it has not been
/// installed yet (tests / unit paths), falls back to the 2 GiB default.
pub(crate) fn active_ram_cap_bytes() -> u64 {
    RAM_CAP_BYTES
        .get()
        .copied()
        .unwrap_or_else(|| resolve_ram_cap_bytes(None))
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Running hit/miss/eviction counters for a `PromptCache` instance.
///
/// Updated by `find_best_prefix` (hit/miss) and `push` (evictions).
/// `bytes` is computed lazily on `PromptCache::stats()` — it is NOT
/// kept incrementally because slots can be truncated/evicted at any time.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of times `find_best_prefix` returned a match (exact or prefix).
    pub hits: u64,
    /// Number of times `find_best_prefix` returned `None`.
    pub misses: u64,
    /// Number of slots evicted (LRU policy: slot-count cap or RAM-bytes cap).
    pub evictions: u64,
    /// Approximate total RAM held by all occupied slots, in bytes.
    ///
    /// Populated when `PromptCache::stats()` is called; zero otherwise.
    pub bytes: u64,
    /// Total 256-token blocks matched across all hit requests.
    pub block_hits: u64,
    /// Total 256-token blocks missed across all requests (hit + miss).
    pub block_misses: u64,
    /// Number of hit requests where at least one block was NOT matched
    /// (i.e., partial prefix hit: best_blocks > 0 && best_blocks < want_blocks).
    pub partial_hits: u64,
    /// Number of RAM-cache misses that were served from the SSD tier.
    ///
    /// Incremented by [`PromptCache::hydrate_from_ssd`] each time a longest-
    /// prefix block was read back from a `.kvb` file and promoted into RAM.
    /// Zero when no SSD source is attached (the RAM-only default).
    pub ssd_hits: u64,
}

// ---------------------------------------------------------------------------
// PromptCacheEntry trait
// ---------------------------------------------------------------------------

/// Contract for a single cached post-prefill snapshot.
///
/// Each arch implements this on its own concrete entry struct. The trait is
/// `pub(crate)` — not exported from the crate — because no external callers
/// need it. `PromptCache<E>` is the only consumer.
pub(crate) trait PromptCacheEntry: Sized {
    /// The full prompt token IDs that produced this snapshot.
    ///
    /// Used at both arch callsites to confirm a true full-token-equality
    /// Exact hit (identical-prompt repeat). C1's block-hash matching narrows
    /// the candidate slot via `find_best_prefix`; this method then verifies
    /// the FULL token sequence is byte-identical before the no-truncate,
    /// no-reprefill Exact fast path is taken. Block-floored equality is
    /// insufficient (it misroutes identical prompts into the unsafe partial
    /// path). Also required by C2/C3 (SSD persistence rehydrate + re-verify).
    fn prompt_token_ids(&self) -> &[u32];

    /// Deep clone of all MLX arrays inside the entry.
    ///
    /// `Array::try_clone` is a refcount increment on the MLX-c side — no
    /// tensor data is copied. The returned entry shares GPU buffers until
    /// a write (prefill of a tail) triggers MLX copy-on-write.
    fn deep_clone(&self) -> Result<Self>;

    /// Chained 256-token block digests of this entry's prompt.
    ///
    /// One digest per full block (trailing partial block excluded), accumulated
    /// at construction via `chained_block_hashes`. Used by
    /// `find_best_prefix` for block-aligned longest-prefix matching.
    fn block_hashes(&self) -> &[u64];

    /// The entry's full-attention KV caches (one per decoder layer).
    ///
    /// Drives the default [`kv_bytes`] body (the mutable counterpart
    /// [`kv_caches_mut`] drives [`truncate_kv_to`]). Pure-attention archs return
    /// their only cache vector; hybrid archs return just the KV half (the
    /// recurrent half is exposed via [`lin_caches`]).
    ///
    /// [`kv_caches_mut`]: PromptCacheEntry::kv_caches_mut
    /// [`truncate_kv_to`]: PromptCacheEntry::truncate_kv_to
    /// [`kv_bytes`]: PromptCacheEntry::kv_bytes
    /// [`lin_caches`]: PromptCacheEntry::lin_caches
    fn kv_caches(&self) -> &[KvCache];

    /// Mutable view of the KV caches, for the default [`truncate_kv_to`] body.
    ///
    /// [`truncate_kv_to`]: PromptCacheEntry::truncate_kv_to
    fn kv_caches_mut(&mut self) -> &mut [KvCache];

    /// The runtime `KvQuant` discriminant in effect when this snapshot was
    /// written, or `None` for the legacy sentinel.
    ///
    /// REQUIRED (not defaulted): a defaulted `None` would silently disable the
    /// SSD-spill key tagging in `SpillSink::spill`, dropping every block.
    ///
    /// Read by the blanket `SpillSink::spill` impl to tag each spilled block
    /// with its codec. Kept on the contract (not field access) so the trait can
    /// drive the generic spill path without per-arch field reach-in.
    fn kv_quant(&self) -> Option<KvQuant>;

    /// True when this entry was reconstructed from the SSD tier rather than
    /// produced by a live prefill.
    ///
    /// An SSD-hydrated entry restores only the block-aligned prefix KV — the
    /// first decode token was never serialized, so `first_id` / `first_piece`
    /// are placeholders (id 0). The exact-hit fast path MUST exclude such an
    /// entry (`!entry.is_ssd_hydrated()`) so it falls through to a full
    /// re-prefill that recomputes the real first token; replaying the
    /// placeholder poisons generation (a sentinel "!" first token).
    ///
    /// Defaults to `false` (a normal RAM-cached entry). Arch entries that can
    /// be hydrated from SSD override this to surface their stored flag. Do NOT
    /// use a `first_id == 0` heuristic as a substitute — `<bos>` is id 0 for
    /// some models.
    fn is_ssd_hydrated(&self) -> bool {
        false
    }

    /// The entry's linear-attention recurrent caches (GDN layers).
    ///
    /// Pure-attention archs return `&[]`; hybrid archs (Qwen3.5-MoE) return
    /// the real per-layer slice. Drives the default [`kv_bytes`] body.
    ///
    /// REQUIRED (not defaulted): a defaulted `&[]` would let a future hybrid
    /// arch silently account/spill KV-only and lose its recurrent state.
    ///
    /// [`kv_bytes`]: PromptCacheEntry::kv_bytes
    fn lin_caches(&self) -> &[LinearAttnCache];

    /// Truncate KV caches to `prefix_len` sequence positions in-place.
    ///
    /// Called on a cloned entry before re-prefilling the tail. The default
    /// body trims only the KV caches (`offset() > 0` guard skips never-filled
    /// layers). The recurrent GDN [`lin_caches`] are deliberately NOT reachable
    /// from this default — that "never truncate linear state" invariant is now
    /// structural: linear state is re-run on the tail, never sliced. Override
    /// only for a mock or a genuinely different per-arch policy.
    ///
    /// [`lin_caches`]: PromptCacheEntry::lin_caches
    fn truncate_kv_to(&mut self, prefix_len: usize) {
        for kv in self.kv_caches_mut() {
            if kv.offset() > 0 {
                kv.truncate_to(prefix_len as i32);
            }
        }
        // lin caches deliberately untouched: recurrent GDN state is re-run on
        // the tail, never truncated. Structural now — the default body cannot
        // reach them.
    }

    /// Block-aligned truncation: trim KV caches to `block_count` full blocks.
    ///
    /// Default delegation to `truncate_kv_to(block_count * BLOCK_TOKENS)`.
    /// The live caller is the gemma4 partial-prefix `CacheLookup::Prefix` path
    /// (`gemma4/generate.rs`), gated on `Gemma4Entry::can_truncate_to_block`
    /// (every layer cache trimmable — no wrapped-SWA desync). Qwen3.5-MoE never
    /// calls it: its recurrent GDN `lin_caches` cannot be reconstructed from a
    /// block-truncated KV, so MoE is gated to full-token-equality (Exact) reuse.
    fn truncate_kv_to_block(&mut self, block_count: usize) {
        self.truncate_kv_to(block_count * BLOCK_TOKENS);
    }

    /// Approximate RAM held by this entry's KV/recurrent state, in bytes.
    ///
    /// Default body sums KV `resident_bytes` plus linear-attn `resident_bytes`
    /// (the latter is empty for pure-attention archs). Used by
    /// `PromptCache::stats()` to populate `CacheStats::bytes` and by the
    /// RAM-cap eviction policy in `push`. Best-effort estimate — does not
    /// guarantee byte-exact Metal accounting.
    fn kv_bytes(&self) -> u64 {
        self.kv_caches()
            .iter()
            .map(KvCache::resident_bytes)
            .sum::<u64>()
            + self
                .lin_caches()
                .iter()
                .map(LinearAttnCache::resident_bytes)
                .sum::<u64>()
    }

    /// True unless a hydrated entry is missing K/V payload for a layer this
    /// architecture attends to.
    ///
    /// Default `true`: a fully RAM-resident snapshot (or a pure-attention arch
    /// whose SSD blocks restore every layer) is always complete. gemma4 SWA
    /// overrides this — its rotating-ring layers are not serialised to the SSD
    /// tier, so a hydrated entry can come back with a payload-less attended
    /// layer; the consume engine excludes such an entry from prefix reuse.
    fn is_hydrate_complete(&self) -> bool {
        true
    }

    /// The reusable-prefix policy seam: decide whether (and how) the entry's
    /// cached prefix may be reused for `prompt_ids`.
    ///
    /// `matched_blocks` is `find_best_prefix`'s block_count for this entry.
    /// `is_ssd_hydrated` mirrors [`is_ssd_hydrated`]. Each arch keeps its own
    /// predicate (they differ deliberately); the engine does not impose one.
    ///
    /// Default `None`: no partial reuse (full-token-equality Exact-only archs,
    /// e.g. qwen3 dense and every ExactOnly retrofit).
    ///
    /// [`is_ssd_hydrated`]: PromptCacheEntry::is_ssd_hydrated
    fn is_reusable_prefix_of(
        &self,
        _prompt_ids: &[u32],
        _is_ssd_hydrated: bool,
        _matched_blocks: usize,
    ) -> Option<ReuseKind> {
        None
    }

    /// Clone the entry for reuse, applying any truncation the `kind` requires.
    ///
    /// Atomic: on `Err` the engine yields `Miss` and the source slot is left
    /// untouched. Default body is a plain [`deep_clone`] (correct for
    /// `StrictPrefix`, where the whole cached prefix is reused verbatim). gemma4
    /// overrides `BlockTruncate` to deep-clone then trim the KV to the block
    /// boundary.
    ///
    /// [`deep_clone`]: PromptCacheEntry::deep_clone
    fn prepare_reuse(&self, _kind: ReuseKind) -> Result<Self> {
        self.deep_clone()
    }
}

// ---------------------------------------------------------------------------
// SpillSink — optional SSD-spill hook
// ---------------------------------------------------------------------------

/// Sink that receives an entry as it is being evicted from RAM, so it can be
/// persisted to the SSD tier (`KvBlockWriter` → `SsdKvIndex`) instead of
/// silently dropped.
///
/// Generic over the entry type so `PromptCache<E>` stays arch-agnostic and the
/// hook is mockable in tests (a test sink can capture the evicted entry's hash
/// without touching disk). The real implementation (`kv_cache::spill::SsdSpiller`)
/// is fire-and-forget: it does a cheap refcount-clone of the entry's caches and
/// hands them to a background drain thread; on any failure it `warn!`s and drops
/// the job — eviction never blocks on disk I/O and never panics.
///
/// `spill` is called with the entry **by reference, before it is dropped**, on
/// both eviction paths (RAM-cap and slot-count). When the cache holds no sink
/// (`None`), eviction is exactly the pre-pure drop.
pub(crate) trait SpillSink<E>: Send {
    /// Spill an entry that is about to be evicted. Must not block or panic.
    fn spill(&self, entry: &E);
}

/// Blanket spill impl over any `PromptCacheEntry`.
///
/// Replaces the per-arch `impl SpillSink<…Entry> for SsdSpiller` blocks: the
/// entry shape is reached only through the trait accessors, so the same body
/// serves pure-attention archs (`lin_caches()` returns `&[]`) and hybrid GDN
/// archs (`lin_caches()` returns the real recurrent slice) identically. Local
/// trait + foreign `SsdSpiller` is a coherent blanket impl; the per-arch
/// blocks are deleted so no overlap is possible.
///
/// Fire-and-forget: entries without a full block or a known `kv_quant` are
/// skipped; any clone/eval error is `warn!`-logged and the job dropped —
/// eviction never blocks on disk I/O and never panics.
impl<E: PromptCacheEntry> SpillSink<E> for SsdSpiller {
    fn spill(&self, entry: &E) {
        let Some(&hash) = entry.block_hashes().last() else {
            return; // no full block → no stable spill key
        };
        let Some(kv_quant) = entry.kv_quant() else {
            return; // unknown quant → cannot tag the block
        };
        let kv: Result<Vec<KvCache>> = entry
            .kv_caches()
            .iter()
            .map(KvCache::try_deep_clone)
            .collect();
        let lin: Result<Vec<LinearAttnCache>> = entry
            .lin_caches()
            .iter()
            .map(LinearAttnCache::try_deep_clone)
            .collect();
        let (kv_caches, lin_caches) = match (kv, lin) {
            (Ok(k), Ok(l)) => (k, l),
            (Err(e), _) | (_, Err(e)) => {
                tracing::warn!(model_id = self.model_id(), error = %e,
                    "kv-spill: cache clone failed, skipping spill");
                return;
            }
        };
        // Eval on the inference thread before handing buffers to the spill worker —
        // lazy arrays must be materialized while the Metal context is ours.
        if let Err(e) = kv_caches
            .iter()
            .try_for_each(KvCache::eval_for_spill)
            .and_then(|()| {
                lin_caches
                    .iter()
                    .try_for_each(LinearAttnCache::eval_for_spill)
            })
        {
            tracing::warn!(model_id = self.model_id(), error = %e,
                "kv-spill: eval failed, skipping spill");
            return;
        }
        self.try_spill(SpillJob {
            hash,
            layout_key: self.layout_key(),
            model_id: self.model_id().to_string(),
            kv_quant,
            kv_caches,
            lin_caches,
        });
    }
}

// ---------------------------------------------------------------------------
// SsdHydrate — lives in `rmlx_kv_ssd::traits`. Re-exported above.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Slot wrapper (internal)
// ---------------------------------------------------------------------------

/// Internal slot: wraps an entry with its LRU sequence number.
pub(crate) struct Slot<E> {
    pub(crate) entry: E,
    /// Monotonically increasing counter set when this slot is last accessed
    /// (either on first `push` or on a `find_best_prefix` hit).
    last_used_seq: u64,
    /// stable monotonic id assigned at first `push`. Used as the
    /// `slot_id` payload stored inside the per-cache [`PrefixIndex`]; the
    /// index hands this back on `match_best`, and the cache scans
    /// `self.slots` to convert it back to a current `usize` index (the
    /// `Vec` index is unstable under `swap_remove`). Cost: O(slots) on a
    /// hit, but `slots.len()` is small (default 4, max realistically 256).
    slot_uid: u64,
}

// ---------------------------------------------------------------------------
// PromptCache<E>
// ---------------------------------------------------------------------------

/// Multi-slot LRU prompt cache, generic over the arch-specific entry type.
///
/// See module-level docs for the slot semantics and eviction policy.
pub(crate) struct PromptCache<E: PromptCacheEntry> {
    pub(crate) slots: Vec<Slot<E>>,
    pub(crate) capacity: usize,
    /// RAM cap in bytes. Set at construction from the CLI flag. Default 2 GiB.
    pub(crate) max_bytes: u64,
    /// Global monotonic sequence counter. Incremented on every
    /// `find_best_prefix` hit and `push`. Never resets (u64 overflow
    /// would require >1.8 × 10^19 cache accesses — irrelevant in practice).
    seq: u64,
    pub(crate) stats: CacheStats,
    /// Optional SSD-spill hook. `None` → evicted entries are dropped
    /// exactly as before. `Some(_)` → each evicted entry is offered to the
    /// sink (fire-and-forget) before being dropped.
    spill: Option<Box<dyn SpillSink<E>>>,
    /// Optional SSD-hydrate source. `None` → a RAM miss is just a miss
    /// (today's behavior). `Some(_)` → on a RAM miss the source is queried for
    /// the longest matching block-hash prefix and, on a hit, the reconstructed
    /// entry is promoted into RAM via [`PromptCache::hydrate_from_ssd`].
    ssd: Option<Box<dyn SsdHydrate<E>>>,
    /// which `PrefixIndex` impl this cache uses for `find_best_prefix`.
    /// Stored alongside the trait object so we can early-out on the Linear
    /// path (preserve the pre-byte-identical scan with no index
    /// upkeep) and only pay index-maintenance cost on the Radix path.
    prefix_index_kind: PrefixIndexKind,
    /// pluggable longest-prefix index. Mirrors `self.slots` —
    /// every entry in `slots` has exactly one corresponding entry here
    /// keyed by `(slot.entry.block_hashes(), layout_key=0)` with payload =
    /// `slot.slot_uid`. Model / layout / codec disambiguation is already baked
    /// into the stored `block_hashes` by the per-arch push path
    /// (`chained_block_hashes_seeded(ids, cache_seed(…))`), so we pass `0` to
    /// the trait and rely on hash inequality alone.
    ///
    /// On the Linear path this index is built + maintained for parity with
    /// the Radix path (used by the differential test in the bench)
    /// but `find_best_prefix` ignores it and walks `slots` directly. On the
    /// Radix path the index is the query target; the slot scan is reduced
    /// to a slot_uid → idx lookup (O(slots), `slots.len()` ≤ 256).
    prefix_index: Box<dyn PrefixIndex>,
    /// monotonic counter handed out as `Slot::slot_uid` on each push.
    next_slot_uid: u64,
}

impl<E: PromptCacheEntry> PromptCache<E> {
    /// Create a new cache with the given slot capacity.
    ///
    /// `capacity == 0` is a real state, not a degenerate one-slot cache: the
    /// cache is built, counts its misses and keeps its SSD sinks, but [`push`]
    /// refuses every entry, so nothing is ever stored and every request
    /// prefills. That is what `--prompt-cache-slots 0` means.
    ///
    /// RAM cap is taken from the process-global resolver
    /// ([`active_ram_cap_bytes`]): CLI `--prompt-cache-ram-gb` if installed,
    /// else default 2 GiB.
    ///
    /// [`push`]: PromptCache::push
    pub(crate) fn new(capacity: usize) -> Self {
        Self::with_max_bytes(capacity, active_ram_cap_bytes())
    }

    /// Create a cache with an explicit RAM cap (used in tests).
    pub(crate) fn with_max_bytes(capacity: usize, max_bytes: u64) -> Self {
        let kind = active_prefix_index_kind();
        Self {
            slots: Vec::with_capacity(capacity),
            capacity,
            max_bytes,
            seq: 0,
            stats: CacheStats::default(),
            spill: None,
            ssd: None,
            prefix_index_kind: kind,
            prefix_index: build_active_index(),
            next_slot_uid: 0,
        }
    }

    /// Attach an SSD-spill sink. Evicted entries are offered to `sink`
    /// before being dropped. Idempotent-overwrite: replaces any prior sink.
    ///
    /// Off by default (`new`/`with_max_bytes` leave it `None`); a model-load
    /// path opts in by calling this once. With no sink, eviction is the
    /// unchanged pre-pure drop.
    ///
    /// wires the production caller: `ssd_tier::attach_at_load` →
    /// per-arch `install_ssd_sinks` spawns a spiller and calls this.
    pub(crate) fn set_spill_sink(&mut self, sink: Box<dyn SpillSink<E>>) {
        self.spill = Some(sink);
    }

    /// Attach an SSD-hydrate source (, symmetric to [`set_spill_sink`]).
    /// On a RAM miss the source is queried for the longest matching block-hash
    /// prefix; a hit is reconstructed and promoted into RAM.
    ///
    /// Off by default (`None`) → RAM-only behavior, unchanged from pre-.
    /// wires the production caller at model-load (`ssd_tier::attach_at_load`),
    /// gated by `--kv-ssd-cache-gb`.
    ///
    /// [`set_spill_sink`]: PromptCache::set_spill_sink
    pub(crate) fn set_ssd_source(&mut self, source: Box<dyn SsdHydrate<E>>) {
        self.ssd = Some(source);
    }

    /// On a RAM-cache miss, try to serve the request from the SSD tier.
    ///
    /// Queries the attached [`SsdHydrate`] source for the longest matching
    /// block-hash prefix of `prompt_ids`. On a hit, the reconstructed entry is
    /// `push`ed into RAM (which may spill a different LRU entry via — the
    /// symmetric path) and `stats.ssd_hits` is bumped; the new slot index is
    /// returned. On a miss, or when no SSD source is attached, returns `None`
    /// and the caller falls through to a full prefill.
    ///
    /// Call this only after [`find_best_prefix`] returns `None`, and pass it
    /// the **same** `seed` that call used: the source probes the index with
    /// that value and recomputes the promoted entry's block hashes from it, so
    /// the retried `find_best_prefix` matches what was just hydrated. `kv_quant`
    /// is the request's codec, used to verify the block header and tag the
    /// reconstructed entry, and `dispatch_policy` is the kernel-path policy the
    /// reconstructed caches must carry — the source holds none of the three
    /// itself, because it is shared by every model of the arch and every
    /// hot-swapped codec.
    ///
    /// Corruption (bad read / metadata mismatch / missing file) is handled
    /// inside the source impl (delete file + index row, `warn!`) and surfaces
    /// as a miss here — this method never panics.
    ///
    /// [`find_best_prefix`]: PromptCache::find_best_prefix
    pub(crate) fn hydrate_from_ssd(
        &mut self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        dispatch_policy: DispatchPolicy,
    ) -> Option<usize> {
        // A zero-slot cache can admit nothing, so hydrating would read a `.kvb`
        // off disk and reconstruct its K/V only for `push` to refuse it — once
        // per request, for the life of the process. Worse, the refusal would
        // arrive at the over-cap branch below and be logged as "exceeds RAM
        // cap", which is not what happened.
        if self.capacity == 0 {
            return None;
        }
        // Take the source out so the `&self` borrow during `hydrate` does not
        // conflict with the `&mut self` `push` below; put it back after.
        let source = self.ssd.take()?;
        let result = source.hydrate(prompt_ids, seed, kv_quant, dispatch_policy);
        self.ssd = Some(source);
        match result {
            Ok(Some(entry)) => {
                if let Some(idx) = self.push(entry) {
                    self.stats.ssd_hits += 1;
                    Some(idx)
                } else {
                    // Reconstructed block's KV alone exceeds the RAM cap, so it
                    // was refused admission — a hydrate that cannot be resident
                    // is a miss; the caller re-prefills.
                    tracing::debug!(
                        prompt_len = prompt_ids.len(),
                        "ssd-hydrate: reconstructed block exceeds RAM cap — treating as miss"
                    );
                    None
                }
            }
            Ok(None) => None,
            Err(e) => {
                // The source contract maps corruption to Ok(None); a residual
                // Err is logged and treated as a miss — never propagated.
                tracing::warn!(error = %e, "ssd-hydrate: source error, treating as miss");
                None
            }
        }
    }

    /// Return accumulated hit/miss/eviction counters plus current byte estimate.
    ///
    /// `bytes` is computed fresh on each call by summing `kv_bytes()` over all
    /// occupied slots — no incremental bookkeeping, so this is always accurate
    /// without worrying about eviction/truncation invalidating a running total.
    pub(crate) fn stats(&self) -> CacheStats {
        let bytes: u64 = self.slots.iter().map(|s| s.entry.kv_bytes()).sum();
        CacheStats {
            bytes,
            ..self.stats.clone()
        }
    }

    /// Clear all RAM slots and reset stats.
    /// Called when model is unloaded or capacity changes.
    ///
    /// RAM only. An attached SSD source ([`Self::set_ssd_source`]) survives, so
    /// the next lookup can still miss in RAM and be served by
    /// [`Self::hydrate_from_ssd`] — which bumps `stats.ssd_hits`, not
    /// `stats.hits`. This is not a guarantee that the next request prefills.
    pub(crate) fn clear(&mut self) {
        self.slots.clear();
        self.stats = CacheStats::default();
        self.seq = 0;
        self.prefix_index.clear();
        self.next_slot_uid = 0;
    }

    /// Find the slot with the longest block-aligned prefix match.
    ///
    /// Computes the incoming prompt's chained 256-token block hashes, then for
    /// each slot counts the number of leading block digests that match.
    /// Returns `Some((slot_index, matched_block_count))` for the best match
    /// with `matched_block_count >= 1` (the 256-token worthwhile floor), or
    /// `None` if no slot shares at least one full block.
    ///
    /// Because the digests are chained (block N folds in block N-1), a run of
    /// `k` leading equal digests proves the entire `k * 256`-token prefix is
    /// byte-identical — no token-level rescan needed.
    ///
    /// On a hit, bumps `last_used_seq` of the winning slot to mark it MRU.
    /// Increments `stats.hits` on a match, `stats.misses` on `None`.
    ///
    /// Eviction is NOT performed here — call `push` afterwards if needed.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn find_best_prefix(
        &mut self,
        prompt_ids: &[u32],
        seed: u64,
    ) -> Option<(usize, usize)> {
        // `seed` salts the query digest stream so it partitions by the same
        // `(model, layout, codec)` the push side seeds with — see `cache_seed`.
        // A different model of this arch, a different KV codec, or a different
        // SSD layout produces a disjoint digest stream for the same tokens and
        // never matches another's slots. Pass `FNV_OFFSET` for the legacy
        // un-salted stream (tests only).
        let want = chained_block_hashes_seeded(prompt_ids, seed);

        let (best_idx, best_blocks) = match self.prefix_index_kind {
            // Linear path is byte-identical to the pre-scan;
            // the parallel `prefix_index` is maintained (so the bench
            // harness can swap it in) but unused on lookup.
            PrefixIndexKind::Linear => {
                let mut best_idx: Option<usize> = None;
                let mut best_blocks = 0usize; // hit requires >= 1 block
                for (i, slot) in self.slots.iter().enumerate() {
                    let have = slot.entry.block_hashes();
                    let matched = want
                        .iter()
                        .zip(have.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    if matched > best_blocks {
                        best_blocks = matched;
                        best_idx = Some(i);
                    }
                }
                (best_idx, best_blocks)
            }
            // Radix path — query the tree, then convert the returned
            // `slot_uid` payload back to a slot index via a linear scan
            // (small N, dominated by the radix lookup's O(n_blocks) cost).
            PrefixIndexKind::Radix => match self.prefix_index.match_best(&want, 0) {
                Some((slot_uid, blocks)) => {
                    let idx = self.slots.iter().position(|s| s.slot_uid == slot_uid);
                    // Stale uid (defensive — index/slots desync should be
                    // unreachable but we fall back to Linear scan instead
                    // of panicking).
                    if let Some(i) = idx {
                        (Some(i), blocks)
                    } else {
                        tracing::warn!(
                            slot_uid,
                            "radix returned a stale slot_uid; falling back to linear scan"
                        );
                        let mut best_idx: Option<usize> = None;
                        let mut best_blocks = 0usize;
                        for (i, slot) in self.slots.iter().enumerate() {
                            let matched = want
                                .iter()
                                .zip(slot.entry.block_hashes().iter())
                                .take_while(|(a, b)| a == b)
                                .count();
                            if matched > best_blocks {
                                best_blocks = matched;
                                best_idx = Some(i);
                            }
                        }
                        (best_idx, best_blocks)
                    }
                }
                None => (None, 0),
            },
        };

        let want_blocks = want.len();
        if let Some(i) = best_idx {
            // Bump MRU: advance seq and stamp the winning slot.
            self.seq += 1;
            self.slots[i].last_used_seq = self.seq;
            self.stats.hits += 1;
            self.stats.block_hits += best_blocks as u64;
            self.stats.block_misses += (want_blocks - best_blocks) as u64;
            if best_blocks > 0 && best_blocks < want_blocks {
                self.stats.partial_hits += 1;
            }
        } else {
            self.stats.misses += 1;
            self.stats.block_misses += want_blocks as u64;
        }

        best_idx.map(|i| (i, best_blocks))
    }

    /// Push a new entry, evicting LRU slot(s) as needed.
    ///
    /// Returns `Some(slot_index)` of the newly-stored slot, or `None` when the
    /// entry was refused admission (its own KV alone exceeds the RAM cap — see
    /// the over-cap guard below).
    ///
    /// Admission + eviction order:
    /// 0. Zero-capacity guard: a cache with no slots stores nothing — refuse
    ///    admission (return `None`).
    /// 1. Over-cap guard: if the incoming entry's KV alone exceeds `max_bytes`,
    ///    refuse admission (return `None`) WITHOUT evicting any existing slot.
    /// 2. While `total_kv_bytes + new_entry_bytes > max_bytes` and slots
    ///    remain: evict the LRU slot.
    /// 3. If `slots.len() == capacity`: drop the slot with the smallest
    ///    `last_used_seq`.
    ///
    /// Either cap can trigger independently. Each eviction increments
    /// `stats.evictions`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(crate) fn push(&mut self, entry: E) -> Option<usize> {
        // --- Zero-capacity guard ---
        // `capacity == 0` is the disabled cache. Refusing here is what makes it
        // disabled: `slots` then stays empty for the process's lifetime, so
        // `find_best_prefix` can only miss and every request prefills. It also
        // keeps the slot-count eviction below from asking an empty `slots` for
        // its LRU index.
        if self.capacity == 0 {
            tracing::debug!(
                "prompt cache: zero slots configured — snapshot not stored, next request prefills"
            );
            return None;
        }

        let new_bytes = entry.kv_bytes();

        // --- Over-cap admission guard ---
        // A single snapshot whose resident KV alone exceeds the cap must not be
        // admitted. Admitting it violates the cap and, worse, the next warm
        // (Exact) hit deep-clones the slot and copy-on-writes the full-size KV
        // on the first decode append — doubling peak residency and stalling
        // decode. Refusing admission bounds peak memory to one live copy; the
        // repeat request simply re-prefills, exactly like the cold request.
        // Existing (smaller, valid) slots are left intact — evicting them to
        // make room for an entry that still could not fit would only thrash.
        if new_bytes > self.max_bytes {
            tracing::warn!(
                entry_bytes = new_bytes,
                max_bytes = self.max_bytes,
                n_slots = self.slots.len(),
                "prompt cache: snapshot KV exceeds RAM cap — not admitted; \
                 raise --prompt-cache-ram-gb to cache this context"
            );
            return None;
        }

        // --- RAM-cap eviction ---
        let mut total: u64 = self.slots.iter().map(|s| s.entry.kv_bytes()).sum();
        while !self.slots.is_empty() && total + new_bytes > self.max_bytes {
            let lru_idx = self.lru_slot_index();
            total -= self.slots[lru_idx].entry.kv_bytes();
            let evicted = self.slots.swap_remove(lru_idx);
            // drop the evicted entry from the prefix index BEFORE
            // the entry is offered to the SSD spill sink — keeps the index
            // ⇔ slots invariant intact even if `spill` later panics.
            self.prefix_index.remove(evicted.entry.block_hashes(), 0);
            self.spill_evicted(&evicted.entry);
            self.stats.evictions += 1;
        }

        // --- Slot-count-cap eviction ---
        if self.slots.len() == self.capacity {
            let lru_idx = self.lru_slot_index();
            let evicted = self.slots.swap_remove(lru_idx);
            self.prefix_index.remove(evicted.entry.block_hashes(), 0);
            self.spill_evicted(&evicted.entry);
            self.stats.evictions += 1;
        }

        // Stamp the new slot with the next sequence number + stable uid.
        self.seq += 1;
        self.next_slot_uid += 1;
        let slot_uid = self.next_slot_uid;
        // index the new entry's block-hash chain before pushing.
        self.prefix_index.insert(entry.block_hashes(), 0, slot_uid);
        self.slots.push(Slot {
            entry,
            last_used_seq: self.seq,
            slot_uid,
        });
        Some(self.slots.len() - 1)
    }

    /// Evict the slot at `idx` and bump `stats.evictions`.
    ///
    /// Used by call-site lookup logic that detects an entry mismatch (e.g.
    /// stored `KvQuant` differs from the runtime `KvQuant` — Plan §D8) and
    /// needs to drop the unusable slot before falling back to a Miss. Uses
    /// `swap_remove` so it is O(1); slot order is not meaningful for any
    /// other cache invariant.
    ///
    /// Panics if `idx` is out of range.
    pub(crate) fn evict_slot(&mut self, idx: usize) {
        let removed = self.slots.swap_remove(idx);
        // keep the prefix index in lockstep with the slot vec.
        self.prefix_index.remove(removed.entry.block_hashes(), 0);
        self.stats.evictions += 1;
    }

    /// Offer an about-to-be-dropped entry to the SSD-spill sink, if attached.
    ///
    /// No-op when no sink is configured (`None`) — the pre-pure-drop
    /// behavior. The sink is contractually fire-and-forget (non-blocking,
    /// never panics), so this stays on the hot path with no I/O.
    fn spill_evicted(&self, entry: &E) {
        if let Some(sink) = &self.spill {
            sink.spill(entry);
        }
    }

    /// Index of the slot with the smallest `last_used_seq` (LRU).
    ///
    /// Panics if `slots` is empty — callers must guard.
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    fn lru_slot_index(&self) -> usize {
        self.slots
            .iter()
            .enumerate()
            .min_by_key(|(_, s)| s.last_used_seq)
            .map(|(i, _)| i)
            .expect("lru_slot_index: slots must be non-empty")
    }
}

// ---------------------------------------------------------------------------
// Consumed / ReuseKind — outcome of the shared consume engine
// ---------------------------------------------------------------------------

/// Outcome of a prompt-cache consume decision.
///
/// Read-only — the engine that produces this never `push`es. The arch maps each
/// variant onto its own decode path: `Exact` skips prefill and replays the
/// stored first token; `Reuse` restores the cloned prefix and the arch forwards
/// the tail; `Miss` falls back to a full re-prefill. The cloned `E` carries
/// whatever per-arch extras the entry holds (e.g. qwen3's `first_logprobs`).
pub(crate) enum Consumed<E> {
    /// Full-token-equality hit: the cloned entry's prompt equals the request
    /// prompt and it is not an SSD placeholder. Reuse the snapshot as-is.
    Exact(E),
    /// Prefix reuse: the cloned entry covers a leading prefix of the request
    /// prompt; the arch re-prefills the remaining tail on top of it.
    ///
    /// Only constructed for `Partial`-policy arches and the SSD-hydrated
    /// strict-prefix case; the dense ExactOnly arch never produces it, so it
    /// reads as dead until a prefix-reusing arch is routed through the engine.
    #[allow(dead_code)]
    Reuse { entry: E, kind: ReuseKind },
    /// No usable entry — re-prefill from scratch.
    Miss,
}

/// How a `Consumed::Reuse` prefix relates to the request prompt.
///
/// Constructed only by prefix-reusing arches (a `StrictPrefix` for the
/// SSD-hydrated strict-prefix / SWA-snapshot case, a `BlockTruncate` for the
/// block-aligned gemma4 path). The dense ExactOnly arch never builds either, so
/// the variants read as dead until such an arch is routed through the engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum ReuseKind {
    /// Reuse the entry's full cached prefix verbatim. `prefix_len` is the
    /// cached token length (NOT necessarily block-aligned). The arch forwards
    /// `prompt_ids[prefix_len..]`. gemma4 B1 (SWA snapshot/restore), qwen3.5-moe
    /// HydratedTail.
    StrictPrefix { prefix_len: usize },
    /// Block-aligned truncate then tail re-prefill (gemma4 only). The reuse
    /// prefix is `effective_blocks * BLOCK_TOKENS` tokens; `prepare_reuse`
    /// truncates the cloned KV to that boundary and asserts the invariant.
    BlockTruncate { effective_blocks: usize },
}

// ---------------------------------------------------------------------------
// ReusePolicy — per-arch hard runtime gate
// ---------------------------------------------------------------------------

/// Per-arch policy gating which kinds of cache reuse are allowed.
///
/// hard runtime gate (NOT a comment): every arch instantiates an
/// [`ArchPromptCache`] with one of these variants. The policy is queried by
/// the generate loop's lookup logic to decide whether a partial / strict-prefix
/// hit may be taken or must degrade to a Miss.
///
/// - [`ReusePolicy::Partial`] — pure-attention arch with trimmable KV (Gemma4).
///   Block-aligned partial-prefix reuse + B1 strict-prefix SWA snapshot/restore
///   are both legal. The arch still has to gate per-slot via
///   `Gemma4Entry::can_truncate_to_block` (SWA ring-buffer wrap).
/// - [`ReusePolicy::ExactOnly`] — full-token-equality reuse only. Block-aligned
///   partial hits MUST degrade to a Miss. Required for any arch that holds
///   non-truncatable state alongside the attention KV — today the only
///   production user is **Qwen3.5-MoE** (recurrent GatedDeltaNet `lin_caches`
///   cannot be reconstructed from a block-truncated prefix). Qwen3-dense also
///   uses this for simplicity (its Exact-hit warm-TTFT case is the dominant
///   one and the bench winner; the partial path was never worth the complexity
///   for an arch with no Prefix consumers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReusePolicy {
    /// Block-aligned partial-prefix reuse allowed (Gemma4 — full-attn + SWA).
    Partial,
    /// Full-token-equality reuse only; partial hits must degrade to Miss
    /// (Qwen3-MoE — GDN recurrent state cannot be block-truncated).
    ExactOnly,
}

// ---------------------------------------------------------------------------
// ArchPromptCache — per-arch shell over `PromptCache<E>`
// ---------------------------------------------------------------------------

/// per-arch SSD-attach parameters held outside the cache so they
/// survive `ensure_prompt_cache`'s capacity-change re-creation.
#[derive(Clone)]
pub(crate) struct AttachParams {
    pub(crate) namespace: String,
    pub(crate) layout_key: u64,
    pub(crate) device: rmlx_mlx::Device,
}

/// unified per-arch wrapper around `PromptCache<E>`.
///
/// Collapses the duplicated static state that every arch's `prompt_cache.rs`
/// used to keep (its `Mutex<Option<PromptCache<E>>>`, `Mutex<Option<Attach…>>`,
/// plus the matching `ensure` / `attach` / `read_stats` boilerplate) into one
/// generic type. Each arch keeps a single
/// `static PROMPT_CACHE: ArchPromptCache<MyEntry> = ArchPromptCache::new(…)`.
///
/// The resident-KV byte counter is deliberately NOT here: it is per model
/// *instance* ([`crate::kv_bytes::KvBytesCounter`], a field on each arch's model
/// struct), because two models of the same arch sharing this static would
/// cross-attribute each other's byte totals.
///
/// The genuinely per-arch parts that stay outside this struct:
/// - the `Entry` struct (Gemma4 = KV only, Qwen3 = KV only, Qwen3.5-MoE =
///   KV + LinearAttn);
/// - the `impl SsdHydrate<Entry> for SsdHydrator` block (Entry-shape specific —
///   reconstruction needs the concrete entry's extra fields). The spill side is
///   the blanket `impl SpillSink<E> for SsdSpiller` above, shared by every arch.
/// - the in-`generate.rs` `CacheLookup` match (Exact / Prefix / Miss), which
///   queries [`ArchPromptCache::policy`] to enforce [`ReusePolicy::ExactOnly`]
///   as a hard runtime check.
///
/// Risk-mitigation contract:
/// - The constructor takes a `ReusePolicy`. The `policy()` getter is the
///   single source of truth — the gate is enforced at the call site
///   (`generate.rs`), not as a comment.
/// - Gemma4-SWA needs the snapshot/restore hook in the entry impl
///   (`is_strict_prefix_of`, `can_truncate_to_block`). The compile-time bound
///   on `attach_ssd_tier` (`SsdSpiller: SpillSink<E>`) keeps SSD attach only
///   reachable for archs that implement the trait — never silently disabled.
pub(crate) struct ArchPromptCache<E: PromptCacheEntry> {
    /// Human-readable arch tag used in tracing events.
    arch_name: &'static str,
    /// Reuse policy: `Partial` (Gemma4) or `ExactOnly` (Qwen3, Qwen3.5-MoE).
    policy: ReusePolicy,
    /// The actual prompt cache. `None` until `ensure_prompt_cache` builds it.
    inner: std::sync::Mutex<Option<PromptCache<E>>>,
    /// SSD-tier attach parameters (set once by `attach_ssd_tier`); replayed on
    /// every cache re-creation so a capacity bump never silently drops the
    /// tier.
    attach: std::sync::Mutex<Option<AttachParams>>,
}

impl<E: PromptCacheEntry> ArchPromptCache<E> {
    /// Construct an empty per-arch cache. `const fn` so each arch can declare
    /// it as a `static`.
    pub(crate) const fn new(arch_name: &'static str, policy: ReusePolicy) -> Self {
        Self {
            arch_name,
            policy,
            inner: std::sync::Mutex::new(None),
            attach: std::sync::Mutex::new(None),
        }
    }

    /// Human-readable arch tag used in tracing events.
    #[allow(dead_code)]
    pub(crate) const fn arch_name(&self) -> &'static str {
        self.arch_name
    }

    /// The arch's reuse policy. Generate-loop call sites query this to enforce
    /// `ExactOnly` as a hard runtime gate (see [`ReusePolicy`]).
    pub(crate) const fn policy(&self) -> ReusePolicy {
        self.policy
    }

    /// Lock the inner cache for the duration of the closure. Centralises the
    /// poison-recovery pattern so call sites do not repeat
    /// `lock().unwrap_or_else(|p| p.into_inner())`.
    ///
    /// On a previously poisoned `Mutex`, the inner state is silently recovered
    /// (matches `PromptCache` policy elsewhere) — a panic inside `f` poisons the
    /// lock for subsequent calls but never propagates.
    pub(crate) fn with_inner_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Option<PromptCache<E>>) -> R,
    {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        f(&mut guard)
    }

    /// Ensure the cache is initialised with `capacity` slots. If the existing
    /// cache already has the same capacity, this is a no-op. Otherwise the
    /// cache is rebuilt (old snapshots discarded) and the SSD spiller /
    /// hydrator are re-installed from the recorded `attach` params (no-op when
    /// the tier is OFF).
    ///
    /// The comparison is against what the cache actually stores, so it holds
    /// for every value including `0`. It has to: this runs once per generation
    /// on every arch, and a capacity that never compares equal would rebuild
    /// (and reset the hit/miss counters, and re-install the SSD sinks) on every
    /// request, which reads as "disabled" while in fact discarding a freshly
    /// built cache each time. `0` disables the cache by refusing admission —
    /// see [`PromptCache::new`] — not by failing to match itself.
    ///
    /// `SsdSpiller: SpillSink<E>` and `SsdHydrator: SsdHydrate<E>` bound the
    /// call so only archs with both trait impls reach this method.
    pub(crate) fn ensure(&self, capacity: usize)
    where
        SsdSpiller: SpillSink<E>,
        SsdHydrator: SsdHydrate<E>,
    {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match guard.as_ref() {
            Some(c) if c.capacity == capacity => {}
            _ => {
                let mut cache = PromptCache::new(capacity);
                let attach = self
                    .attach
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone();
                if let Some(p) = attach {
                    install_ssd_sinks::<E>(&mut cache, &p);
                }
                *guard = Some(cache);
            }
        }
    }

    /// Drop every cached snapshot and reset the hit/miss counters, keeping the
    /// cache object itself — its capacity and its installed SSD sinks — in
    /// place. A no-op when no generation has built the cache yet.
    ///
    /// This is the way to make the next request a guaranteed miss while keeping
    /// the cache configured as production has it — which is what a measurement
    /// wants. Reconfiguring to zero slots also misses every time, but it
    /// measures a different cache than the one being served.
    pub(crate) fn clear(&self) {
        self.with_inner_mut(|slot| {
            if let Some(cache) = slot.as_mut() {
                cache.clear();
            }
        });
    }

    /// Wire the spiller + hydrator onto this arch's prompt cache
    /// for `namespace` at `kv_quant` (/ ). Records the attach params
    /// so they survive a later cache re-creation, then installs the sinks on
    /// the live cache when one already exists.
    ///
    /// These parameters describe the on-disk **store** — which namespace, which
    /// layout the rows in it carry. They deliberately do not carry the
    /// attaching model's identity or codec: this is one slot per architecture
    /// and the last attach wins, so a per-model value recorded here would be
    /// wrong for every other resident model of the arch. Per-request identity
    /// reaches the hydrator through [`Self::consume`] instead.
    pub(crate) fn attach_ssd_tier(
        &self,
        namespace: &str,
        kv_quant: KvQuant,
        layout_key: u64,
        device: rmlx_mlx::Device,
    ) where
        SsdSpiller: SpillSink<E>,
        SsdHydrator: SsdHydrate<E>,
    {
        // `kv_quant` is logged, not stored: it is the launch codec, and any
        // request may hot-swap it. Storing it would put an attach-time codec
        // where a per-request one belongs.
        let params = AttachParams {
            namespace: namespace.to_string(),
            layout_key,
            device,
        };
        *self
            .attach
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(params.clone());
        if let Some(cache) = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            install_ssd_sinks::<E>(cache, &params);
        }
        tracing::info!(
            arch = self.arch_name,
            namespace,
            ?kv_quant,
            layout_key = format!("{layout_key:016x}"),
            "SSD tier attached to prompt cache"
        );
    }

    /// active SSD-tier `layout_key`, or `0` when the tier is OFF.
    /// `FNV_OFFSET ^ 0 == FNV_OFFSET` ⇒ legacy un-salted digests on RAM-only
    /// runs, preserving byte-identical behaviour.
    pub(crate) fn active_layout_key(&self) -> u64 {
        self.attach
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map_or(0, |p| p.layout_key)
    }

    /// Resolve the prompt-cache consume decision for one request.
    ///
    /// Read-only with respect to slot population — never `push`es. It owns the
    /// logic that was duplicated byte-for-byte across the cached arches, AFTER
    /// the caller's `n_tokens == 0` early return (consume is never called for an
    /// empty prompt). The flow:
    ///
    /// 1. `has_image` → `Miss` immediately, WITHOUT consulting the cache (no
    ///    find, no evict side-effect). The cached K/V is keyed by token ids
    ///    only; an image prompt's K/V depends on scattered vision features, so
    ///    it must never be served from or stored under a token-id key.
    /// 2. Compute the partitioned seed via [`request_cache_seed`] from
    ///    `active_layout_key()` + the KV codec + the per-layer mixture that
    ///    codec resolves to + `model_sig`, so a slot stored by a different
    ///    model, under a different codec, under a different per-layer policy,
    ///    or for a different cache layout never cross-serves.
    /// 3. `find_best_prefix` (capturing `block_count`) + the hydrate-from-SSD
    ///    retry (no-op when the tier is OFF).
    /// 4. Quant-mismatch guard: a stored `KvQuant` that differs from the runtime
    ///    quant (or a legacy `None`) means the cached K/V layout is incompatible
    ///    — evict the slot, `warn!`, degrade to `Miss`.
    /// 5. Exact arm: `!is_ssd_hydrated() && prompt_token_ids() == prompt_ids` →
    ///    `Exact(deep_clone())`. The clone preserves whatever per-arch extras the
    ///    entry holds (e.g. qwen3's `first_logprobs`).
    /// 6. Reuse gate ([`ReusePolicy`] + `is_hydrate_complete`): a hydrated
    ///    strict-prefix is permitted under any policy; a non-hydrated partial
    ///    match only under `Partial`. When permitted, the entry's
    ///    `is_reusable_prefix_of` hook decides the [`ReuseKind`]; `prepare_reuse`
    ///    clones (+ truncates) for reuse. Hook `None` or an incomplete hydrate
    ///    degrades to `Miss`.
    ///
    /// Every degrade branch emits exactly one `debug!{branch, reason}` so the
    /// decision is reconstructable from a single run's log — the cached arches
    /// previously had silent degrade arms.
    ///
    /// `n_layers` is the asking model's decoder-layer count — it sizes the
    /// per-layer mixture folded into the seed, so it must be the same count the
    /// caller builds its `KvCache` vector at.
    ///
    /// `model_sig` identifies the model asking. It is not decoration: this
    /// cache is one static per *architecture*, so it is shared by every model
    /// of that arch the process has resident, and the token-id equality the
    /// `Exact` arm checks cannot tell two models' identical prompts apart.
    pub(crate) fn consume(
        &self,
        prompt_ids: &[u32],
        kv_quant: KvQuant,
        n_layers: usize,
        has_image: bool,
        model_sig: u64,
    ) -> Consumed<E> {
        // (1) Image prompts bypass the cache entirely — no find, no evict.
        if has_image {
            tracing::debug!(
                arch = self.arch_name,
                branch = "has_image",
                reason = "image prompt bypasses the token-id-keyed cache",
                "prompt cache consume: Miss (image)"
            );
            return Consumed::Miss;
        }

        // (2) Partitioned seed: identical to the per-arch push seed, so a slot
        // stored by a different model, or under a different KV codec / layout,
        // never matches.
        let seed = request_cache_seed(self.active_layout_key(), kv_quant, n_layers, model_sig);
        let policy = self.policy;
        let arch = self.arch_name;
        // Kernel-path policy for the caches a hydrate would reconstruct. The
        // arches build their live caches from the same process default (see
        // `KvCache::with_quant_max_seq`), so reading it here keeps a hydrated
        // set and a freshly built one in agreement. This is the single seam a
        // per-request policy would replace; everything below it already carries
        // the value instead of re-reading a global.
        let dispatch_policy = rmlx_core::dispatch_policy();

        self.with_inner_mut(|guard| match guard.as_mut() {
            Some(cache) => Self::decide_locked(
                arch,
                policy,
                cache,
                prompt_ids,
                kv_quant,
                seed,
                dispatch_policy,
            ),
            None => Consumed::Miss,
        })
    }

    /// The find → hydrate-retry → quant-guard → Exact → reuse-gate decision,
    /// run while holding the cache lock. Split out of [`consume`] so the single
    /// cohesive decision keeps one home (the `cognitive_complexity` allow scopes
    /// to it, not the lock-management wrapper).
    ///
    /// [`consume`]: ArchPromptCache::consume
    #[allow(
        clippy::indexing_slicing,
        reason = "slot_idx is returned by find_best_prefix as a valid index into `cache.slots` and is not mutated before use"
    )]
    #[allow(
        clippy::cognitive_complexity,
        reason = "one cohesive consume decision (find/guard/exact/reuse arms) — splitting the arms across helpers would scatter a single decision and hurt readability"
    )]
    fn decide_locked(
        arch: &'static str,
        policy: ReusePolicy,
        cache: &mut PromptCache<E>,
        prompt_ids: &[u32],
        kv_quant: KvQuant,
        seed: u64,
        dispatch_policy: DispatchPolicy,
    ) -> Consumed<E> {
        // (3) RAM find, then SSD-hydrate retry. A hydrate hit promotes the
        // block into RAM; re-run find_best_prefix so the promoted slot is
        // matched + quant-checked by the path below.
        let mut raw_match = cache.find_best_prefix(prompt_ids, seed);
        if raw_match.is_none()
            && cache
                .hydrate_from_ssd(prompt_ids, seed, kv_quant, dispatch_policy)
                .is_some()
        {
            raw_match = cache.find_best_prefix(prompt_ids, seed);
        }

        // (4) Quant-mismatch guard. The snapshot is only safe to
        // reuse when the stored KvQuant equals the runtime quant.
        let (slot_idx, block_count) = match raw_match {
            Some((slot_idx, block_count)) => {
                let stored = cache.slots[slot_idx].entry.kv_quant();
                if stored == Some(kv_quant) {
                    (slot_idx, block_count)
                } else {
                    tracing::debug!(
                        arch,
                        branch = "quant_mismatch",
                        reason = "stored KV quant differs from runtime — evict + re-prefill",
                        stored = ?stored,
                        runtime = ?kv_quant,
                        prompt_len = prompt_ids.len(),
                    );
                    tracing::warn!(
                        stored = ?stored,
                        runtime = ?kv_quant,
                        prompt_len = prompt_ids.len(),
                        "prompt cache KV quant mismatch — evicting entry, \
                         degrading to re-prefill"
                    );
                    cache.evict_slot(slot_idx);
                    return Consumed::Miss;
                }
            }
            None => return Consumed::Miss,
        };

        let entry = &cache.slots[slot_idx].entry;
        let is_ssd_hydrated = entry.is_ssd_hydrated();

        // (5) Exact arm: full token-id equality AND a real (non-placeholder)
        // first decode token. An SSD-hydrated entry is excluded — its
        // first_id is the placeholder 0, replaying it poisons generation.
        if !is_ssd_hydrated && entry.prompt_token_ids() == prompt_ids {
            return match entry.deep_clone() {
                Ok(cloned) => Consumed::Exact(cloned),
                Err(e) => {
                    tracing::debug!(
                        arch,
                        branch = "deep_clone_err",
                        reason = "Exact deep_clone failed — re-prefill (source slot untouched)",
                        error = %e,
                        prompt_len = prompt_ids.len(),
                    );
                    Consumed::Miss
                }
            };
        }

        // (6) Reuse gate. A hydrated strict-prefix is permitted under any
        // policy (moe HydratedTail); a non-hydrated partial match only under
        // `Partial` (gemma4). Either way the hydrated entry must be COMPLETE
        // (gemma4 SWA: a payload-less attended layer is not reusable).
        let reuse_eligible = if is_ssd_hydrated {
            if entry.is_hydrate_complete() {
                true
            } else {
                tracing::debug!(
                    arch,
                    branch = "incomplete_hydrate",
                    reason = "hydrated entry has a payload-less attended layer — re-prefill",
                    prompt_len = prompt_ids.len(),
                );
                return Consumed::Miss;
            }
        } else {
            policy == ReusePolicy::Partial
        };

        if reuse_eligible {
            if let Some(kind) =
                entry.is_reusable_prefix_of(prompt_ids, is_ssd_hydrated, block_count)
            {
                return match entry.prepare_reuse(kind) {
                    Ok(prepared) => Consumed::Reuse {
                        entry: prepared,
                        kind,
                    },
                    Err(e) => {
                        tracing::debug!(
                            arch,
                            branch = "deep_clone_err",
                            reason = "prepare_reuse failed — re-prefill (source slot untouched)",
                            error = %e,
                            prompt_len = prompt_ids.len(),
                        );
                        Consumed::Miss
                    }
                };
            }
            tracing::debug!(
                arch,
                branch = "non_reusable",
                reason = "matched slot is not a reusable prefix for this prompt — re-prefill",
                prompt_len = prompt_ids.len(),
                block_count,
            );
            return Consumed::Miss;
        }

        // A matched non-hydrated partial under a non-Partial policy, or a
        // hydrated entry that declined the strict-prefix hook (e.g. the
        // block-aligned equal-length case) — re-prefill.
        tracing::debug!(
            arch,
            branch = "hydrated_declined_to_exact",
            reason = "matched slot is not Exact-eligible and reuse is not permitted — re-prefill",
            is_ssd_hydrated,
            prompt_len = prompt_ids.len(),
        );
        Consumed::Miss
    }

    /// Read the current hit/miss/bytes stats, or `None` if the cache has not
    /// been initialised yet (no request served for this arch since startup).
    pub(crate) fn read_cache_stats(&self) -> Option<CacheStats> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(PromptCache::stats)
    }
}

/// Shared SSD-sink install routine — replaces the three near-identical per-arch
/// `install_ssd_sinks` helpers. Spiller is unconditional; hydrator-open failure
/// logs a `warn!` and leaves the cache spill-only for the run.
fn install_ssd_sinks<E: PromptCacheEntry>(cache: &mut PromptCache<E>, p: &AttachParams)
where
    SsdSpiller: SpillSink<E>,
    SsdHydrator: SsdHydrate<E>,
{
    cache.set_spill_sink(Box::new(SsdSpiller::spawn(
        &p.namespace,
        p.layout_key,
        p.device,
    )));
    match SsdHydrator::open(&p.namespace, p.layout_key, p.device) {
        Ok(h) => cache.set_ssd_source(Box::new(h)),
        Err(e) => tracing::warn!(
            namespace = %p.namespace,
            error = %e,
            "SSD hydrator open failed; spill-only for this run"
        ),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "prompt_cache_tests.rs"]
mod prompt_cache_tests;
