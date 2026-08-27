// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! SSD-tier hydrate source for prompt-cache RAM misses.
//!
//! Symmetric to 's [`crate::kv_cache::spill::SsdSpiller`]: where the
//! spiller persists a RAM-evicted entry to a `.kvb` file + `SsdKvIndex` row,
//! [`SsdHydrator`] reads one back on a RAM miss. It is wired onto the per-arch
//! `PromptCache` via `PromptCache::set_ssd_source`; the arch crate supplies the
//! `SsdHydrate<E>` impl that turns the reconstructed caches into a concrete
//! entry (`Gemma4Entry` / `Qwen35MoeEntry`).
//!
//! ## Flow (per RAM miss)
//!
//! 1. Compute the prompt's chained block hashes (`chained_block_hashes_seeded`)
//!    from the `seed` the caller passed in — the very `u64` the RAM cache is
//!    querying under, not a second one computed here. The probe therefore
//!    cannot drift from the key the spill side wrote; there is no second
//!    computation to drift.
//! 2. `SsdKvIndex::lookup_longest_prefix` — the longest cached block-aligned
//!    prefix (longest-first scan of the candidate prefix digests).
//! 3. On a hit: `block_io::read_caches` reads the `.kvb`, verifying the
//!    `model_id` + `kv_quant` header. On success the index row is `touch`ed and
//!    the reconstructed `(Vec<KvCache>, Vec<LinearAttnCache>)` + matched
//!    `prompt_ids` prefix are returned to the arch impl to wrap as an entry.
//! 4. On corrupt / metadata-mismatch / read error: the `.kvb` file and the
//!    index row are deleted, a `warn!` is logged, and the lookup returns a miss
//!    (the prompt-cache falls through to a full prefill). Never panics.
//!
//! Unlike the spiller, the hydrate read happens **on the request thread** (the
//! lookup path is sync), but only on a RAM miss — the cold path that would
//! otherwise pay a full prefill anyway, so the file read is strictly cheaper.
//!
//! Production wiring (construct an `SsdHydrator` at model-load and
//! `set_ssd_source` it onto the per-arch `PROMPT_CACHE`, gated by the same flag
//! as spill) is done by 's `ssd_tier::attach_at_load`.

use std::path::PathBuf;
use std::time::Instant;

use rmlx_core::DispatchPolicy;
use rmlx_metrics::events::{EventRecorder, SsdHydrateEvent};
use rmlx_mlx::Device;

use rmlx_kv_quant::kvcache::KvCache;
use rmlx_kv_quant::linear_attn::LinearAttnCache;
use rmlx_kv_quant::KvQuant;

use crate::block_io::read_caches_timed_with_identity;
use crate::hashing::{chained_block_hashes_seeded, BLOCK_TOKENS};
use crate::hooks::{call_ssd_hydrate_prom_hook, ssd_event_recorder};
use crate::ssd_index::{KvBlockRow, SsdKvIndex};
use crate::traits::ExactReplayMetadata;

/// Whether a request is safe to serve from a stored full-prompt snapshot.
///
/// A longest-full-block index hit is only a candidate. The request must carry
/// every stored token, in order, and may then append a suffix. Equal prompts
/// are accepted for exact replay; strict extensions are accepted for prefix
/// reuse. Truncation and any divergent suffix fail closed.
pub fn prompt_identity_matches(request: &[u32], stored: &[u32]) -> bool {
    request.len() >= stored.len() && request.starts_with(stored)
}

/// A reconstructed SSD block ready to be wrapped as an arch prompt-cache entry.
///
/// The arch's `SsdHydrate<E>` impl turns this into its concrete entry type,
/// recomputing the block hashes from `prompt_ids` (the matched-prefix token
/// IDs) and recording the runtime `kv_quant`.
#[allow(missing_debug_implementations)]
#[allow(
    clippy::exhaustive_structs,
    reason = "promoted: internal SSD-tier bridge struct, was pub(crate); promoted to pub for cross-crate use by arch SsdHydrate impls in rmlx-models; adding a field is a coordinated update across both crates"
)]
pub struct HydratedBlock {
    /// Complete token IDs stored with this snapshot, not a synthesized request
    /// prefix. Legacy records without identity metadata remain unavailable to
    /// this exact-identity path.
    pub prompt_ids: Vec<u32>,
    /// Optional exact first-token replay captured with the stored prompt.
    pub exact_replay: Option<ExactReplayMetadata>,
    /// Reconstructed attention KV caches (one per layer).
    pub kv_caches: Vec<KvCache>,
    /// Reconstructed GDN linear-attention recurrent state (empty for
    /// pure-attention archs).
    pub lin_caches: Vec<LinearAttnCache>,
}

/// SSD-tier hydrate source: owns an [`SsdKvIndex`] + the namespace it reads.
///
/// Deliberately holds **no** per-model or per-request identity. It is installed
/// on a per-*architecture* prompt cache and outlives the model that installed
/// it: several models of one arch can be resident at a time, and the KV codec
/// is chosen per request. Anything this struct remembered about "the model" or
/// "the codec" would be whatever attached last, and seeding a probe from it
/// would silently stop matching for every other model and every hot-swapped
/// codec. The caller passes those facts to [`Self::lookup`] instead.
///
/// What it does own is the namespace: the `SsdKvIndex` and the `.kvb`
/// directory, resolved once at construction, and the `layout_key` stamped on
/// the rows in it. Those are properties of the store, and the spiller writing
/// into that store is installed from the same attach parameters, so the two
/// always agree.
#[allow(missing_debug_implementations)]
pub struct SsdHydrator {
    index: SsdKvIndex,
    // Resolved namespace dir, kept for parity with `SsdSpiller` + test
    // construction; production `lookup` reads the absolute `row.path` from the
    // index, so this is not read on the hot path.
    #[allow(dead_code)]
    dir: PathBuf,
    model_id: String,
    /// stable u64 hash over the arch + KV layout of the namespace's rows.
    /// Pins index lookups under the composite `(hash, layout_key)` PK; it is a
    /// property of the store, matched by the spiller that fills it.
    layout_key: u64,
    device: Device,
}

impl SsdHydrator {
    /// Open the hydrate source for the `model_id` namespace.
    ///
    /// The `.kvb` directory + index DB are resolved via
    /// `paths::kv_cache_dir(model_id)` — the same namespace the spiller
    /// writes to. Returns `Err` only if the index cannot be opened; the caller
    /// (model-load) decides whether to proceed without an SSD tier.
    pub fn open(
        model_id: impl Into<String>,
        layout_key: u64,
        device: Device,
    ) -> rmlx_core::error::Result<Self> {
        let model_id = model_id.into();
        let dir = rmlx_core::paths::kv_cache_dir(&model_id);
        let index = SsdKvIndex::open(&model_id)
            .map_err(|e| rmlx_core::error::Error::Mlx(format!("ssd-hydrate index open: {e}")))?;
        Ok(Self {
            index,
            dir,
            model_id,
            layout_key,
            device,
        })
    }

    /// Test-only: build a hydrator bound to an explicit dir + already-open
    /// index, bypassing `paths::kv_cache_dir` so tests are hermetic.
    #[cfg(test)]
    pub fn with_index(
        model_id: impl Into<String>,
        layout_key: u64,
        device: Device,
        dir: PathBuf,
        index: SsdKvIndex,
    ) -> Self {
        Self {
            index,
            dir,
            model_id: model_id.into(),
            layout_key,
            device,
        }
    }

    /// `<arch>/<snapshot>` identity this hydrator reads under. Symmetric to
    /// `SsdSpiller::model_id`; used by tests + kept for API parity.
    #[allow(dead_code)]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// layout key of the namespace's rows. Pins the `(hash, layout_key)`
    /// composite-PK lookup.
    pub fn layout_key(&self) -> u64 {
        self.layout_key
    }

    /// Index lookup plus the block-hash chain the RAM cache will look the
    /// promoted entry up by, both under the caller's `seed`.
    ///
    /// The recompute reuses the same `seed` value the probe used, so the
    /// hydrated entry is findable by the query that triggered the hydrate by
    /// construction — there is no second seed to keep in step.
    pub fn lookup_seeded(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> rmlx_core::error::Result<Option<(HydratedBlock, Vec<u64>)>> {
        let Some(block) = self.lookup(prompt_ids, seed, kv_quant, policy)? else {
            return Ok(None);
        };
        let hashes = chained_block_hashes_seeded(&block.prompt_ids, seed);
        Ok(Some((block, hashes)))
    }

    /// Look up + reconstruct the longest cached block-aligned prefix of
    /// `prompt_ids`. Returns `Ok(Some(_))` on an SSD hit, `Ok(None)` on a true
    /// miss **or** on corruption (after deleting the bad file + row + `warn!`).
    ///
    /// `seed` is the requesting model's prompt-cache seed, `kv_quant` the
    /// codec the request is running, and `policy` the kernel paths its caches
    /// dispatch through; all three come from the caller, never from this
    /// struct — see the type docs for why.
    ///
    /// Never panics. The arch `SsdHydrate<E>` impl calls this and wraps the
    /// result as its concrete entry. Emits a [`SsdHydrateEvent`] via the
    /// process-global event recorder (if set) with per-phase timing.
    pub fn lookup(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
    ) -> rmlx_core::error::Result<Option<HydratedBlock>> {
        self.lookup_inner(prompt_ids, seed, kv_quant, policy, None)
    }

    /// Test-only: like [`lookup`] but uses an explicit `EventRecorder` for
    /// hermetic event-capture tests, bypassing the process-global OnceLock.
    #[cfg(test)]
    pub fn lookup_with_recorder(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
        recorder: &EventRecorder,
    ) -> rmlx_core::error::Result<Option<HydratedBlock>> {
        self.lookup_inner(prompt_ids, seed, kv_quant, policy, Some(recorder))
    }

    /// Core of [`lookup`]. `test_recorder` overrides the process-global when
    /// `Some`; otherwise falls back to `ssd_event_recorder()`.
    #[allow(
        clippy::cognitive_complexity,
        reason = "hash → SQLite lookup → file read → dequant → touch → event-emit → \
                  prometheus hook: six sequential phases with per-phase timing; \
                  splitting would scatter timing state across fn boundaries"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::manual_let_else,
        reason = "the match folds a Result<Option<T>> through ? and None-return in one expression; \
                  let-else cannot express the ? chain on the same line without restructuring"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "the hydrate path keeps its six timed phases and fail-closed cleanup in one transaction; splitting it would scatter the phase timing and cleanup invariants"
    )]
    fn lookup_inner(
        &self,
        prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        policy: DispatchPolicy,
        test_recorder: Option<&EventRecorder>,
    ) -> rmlx_core::error::Result<Option<HydratedBlock>> {
        // The caller's seed, used as given. It is the same `u64` the RAM query
        // is running and the same one the push side built the stored digests
        // from, so probe and key agree by construction rather than by two
        // computations happening to match. Recomputing it here from state this
        // struct remembered is what made the tier silently 0-hit: every request
        // re-prefills and nothing reports an error.
        let chained = chained_block_hashes_seeded(prompt_ids, seed);
        if chained.is_empty() {
            return Ok(None); // < one full block → nothing indexable
        }

        // ── Phase 1: SQLite prefix lookup ─────────────────────────────────────
        let t0 = Instant::now();
        let (block_count, row) = match self
            .index
            .lookup_longest_prefix(&chained, self.layout_key)
            .map_err(|e| rmlx_core::error::Error::Mlx(format!("ssd-hydrate index lookup: {e}")))?
        {
            Some(hit) => hit,
            None => return Ok(None),
        };
        let dur_lookup_us = t0.elapsed().as_micros() as u64;

        // ── Phases 2–4: file read + dequant + GPU upload (timed inside block_io) ──
        // Verify the block against the *request's* codec. A row whose digest
        // the request matched but whose header codec differs is genuinely
        // anomalous — the codec salt is part of the seed, so a codec-A row
        // cannot produce a codec-B digest. Checking an attach-time codec here
        // instead would reject perfectly good rows written by a hot-swapped
        // request, and delete them as corrupt.
        match read_caches_timed_with_identity(
            &row.path,
            self.device,
            &self.model_id,
            kv_quant,
            policy,
        ) {
            Ok(Some((
                kv_caches,
                lin_caches,
                identity,
                bytes_read,
                dur_read_us,
                dur_dequant_us,
                dur_finalize_us,
            ))) => {
                let prefix_len = block_count * BLOCK_TOKENS;

                // A v2 record carries the complete prompt that produced its
                // cache state. The index hash only identifies the final full
                // block, so it is a candidate rather than proof of identity.
                // Never manufacture identity from the request prefix: doing
                // so would make a divergent SWA tail look reusable.
                let stored_prompt_ids = if let Some(identity) = identity {
                    if identity.prompt_ids.len() < prefix_len
                        || identity.prompt_ids.len() > i32::MAX as usize
                        || !prompt_identity_matches(prompt_ids, &identity.prompt_ids)
                    {
                        // A valid, differently-tailed record is simply not a
                        // hit for this request. It must remain indexed so the
                        // owning prompt can still hydrate it later.
                        tracing::debug!(
                            hash = %row.hash,
                            block_count,
                            stored_prompt_len = identity.prompt_ids.len(),
                            request_prompt_len = prompt_ids.len(),
                            "ssd-hydrate: block hash matched but full prompt identity did not"
                        );
                        return Ok(None);
                    }

                    // The cache offset is part of the state identity. If the
                    // persisted prompt length and reconstructed cache disagree,
                    // promoting it would attach a cache at the wrong absolute
                    // position (especially unsafe for a rotating window).
                    let stored_len = identity.prompt_ids.len() as i32;
                    let has_producer = kv_caches.iter().any(|cache| cache.offset() == stored_len);
                    let offsets_valid = kv_caches
                        .iter()
                        .all(|cache| matches!(cache.offset(), 0) || cache.offset() == stored_len);
                    if !has_producer || !offsets_valid {
                        tracing::warn!(
                            hash = %row.hash,
                            stored_prompt_len = stored_len,
                            "ssd-hydrate: identity length does not match reconstructed cache offset; deleting block"
                        );
                        self.drop_row(&row);
                        let _ = std::fs::remove_file(&row.path);
                        return Ok(None);
                    }
                    (identity.prompt_ids, identity.exact_replay)
                } else {
                    tracing::warn!(
                        hash = %row.hash,
                        "ssd-hydrate: v2 production block has no full prompt identity; deleting block"
                    );
                    self.drop_row(&row);
                    let _ = std::fs::remove_file(&row.path);
                    return Ok(None);
                };

                // ── Phase 5: SQLite touch (LRU update) ────────────────────────
                let t_touch = Instant::now();
                if let Err(e) = self.index.touch(&row.hash, row.layout_key) {
                    tracing::warn!(
                        hash = %row.hash,
                        layout_key = row.layout_key,
                        error = %e,
                        "ssd-hydrate: touch failed"
                    );
                }
                let dur_touch_us = t_touch.elapsed().as_micros() as u64;
                let dur_us = t0.elapsed().as_micros() as u64;

                tracing::info!(
                    hash = %row.hash,
                    path = %row.path.display(),
                    block_count,
                    prefix_len,
                    bytes_read,
                    dur_us,
                    dur_lookup_us,
                    dur_read_us,
                    dur_dequant_us,
                    dur_finalize_us,
                    dur_touch_us,
                    "ssd-hydrate: RAM miss served from SSD tier"
                );

                // ── Emit per-block event ───────────────────────────────────────
                // Use the injected test recorder if present; else use the global.
                let global_rec;
                let rec_ref: Option<&EventRecorder> = if test_recorder.is_some() {
                    test_recorder
                } else {
                    global_rec = ssd_event_recorder();
                    global_rec.as_deref()
                };

                if let Some(rec) = rec_ref {
                    let ev = SsdHydrateEvent {
                        namespace: self.model_id.clone(),
                        bytes: bytes_read,
                        dur_us,
                        dur_lookup_us,
                        dur_read_us,
                        dur_dequant_us,
                        dur_finalize_us,
                        dur_touch_us,
                        block_count: block_count as u64,
                    };
                    if let Err(e) = rec.record_ssd_hydrate(&ev) {
                        tracing::warn!(
                            hash = %row.hash,
                            error = %e,
                            "ssd-hydrate: event recorder failed (non-fatal)"
                        );
                    }
                }

                // Feed the Prometheus histogram accumulator (registered by
                // rmlx-server at startup via `set_ssd_hydrate_prom_hook`). No-op
                // when unset.
                call_ssd_hydrate_prom_hook(dur_us, bytes_read);

                Ok(Some(HydratedBlock {
                    prompt_ids: stored_prompt_ids.0,
                    exact_replay: stored_prompt_ids.1,
                    kv_caches,
                    lin_caches,
                }))
            }
            Ok(None) => {
                // The block was evicted between this lookup and the read. LRU
                // eviction deletes the row first, so the row is normally gone
                // already; dropping it here covers an evictor that did not get
                // to unlink. Nothing to remove from disk, and nothing wrong.
                tracing::debug!(
                    hash = %row.hash,
                    path = %row.path.display(),
                    "ssd-hydrate: block evicted under the read, falling back to prefill"
                );
                self.drop_row(&row);
                Ok(None)
            }
            Err(e) => {
                // Genuinely bad block (truncated, wrong header, unreadable):
                // delete row + file, fall through to prefill.
                tracing::warn!(
                    hash = %row.hash,
                    path = %row.path.display(),
                    error = %e,
                    "ssd-hydrate: corrupt block, deleting index row + file, falling back to prefill"
                );
                // Row before file, the same order LRU eviction uses: it makes an
                // interrupted cleanup leave an unreferenced row, which
                // `prune_missing` reclaims, instead of an unreferenced file,
                // which nothing does.
                self.drop_row(&row);
                let _ = std::fs::remove_file(&row.path);
                Ok(None)
            }
        }
    }

    /// Drop `row`'s index entry after a read that could not be served.
    fn drop_row(&self, row: &KvBlockRow) {
        if let Err(e) = self.index.delete(&row.hash, row.layout_key) {
            tracing::warn!(
                hash = %row.hash,
                layout_key = row.layout_key,
                error = %e,
                "ssd-hydrate: index row delete failed"
            );
        }
    }
}

#[cfg(test)]
#[path = "hydrate_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "hydrate_identity_tests.rs"]
mod identity_tests;
