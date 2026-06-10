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
//! 1. Compute the prompt's chained block hashes (`chained_block_hashes_seeded`,
//!    seeded with `FNV_OFFSET ^ layout_key ^ kv_quant.cache_key_salt()` — the
//!    spill-side key).
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

use rmlx_metrics::events::{EventRecorder, SsdHydrateEvent};
use rmlx_mlx::Device;

use rmlx_kv_quant::kvcache::KvCache;
use rmlx_kv_quant::linear_attn::LinearAttnCache;
use rmlx_kv_quant::KvQuant;

use crate::block_io::read_caches_timed;
use crate::hashing::{chained_block_hashes_seeded, BLOCK_TOKENS, FNV_OFFSET};
use crate::hooks::{call_ssd_hydrate_prom_hook, ssd_event_recorder};
use crate::ssd_index::SsdKvIndex;

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
    /// Token IDs of the matched block-aligned prefix (`block_count * 256`).
    pub prompt_ids: Vec<u32>,
    /// Reconstructed attention KV caches (one per layer).
    pub kv_caches: Vec<KvCache>,
    /// Reconstructed GDN linear-attention recurrent state (empty for
    /// pure-attention archs).
    pub lin_caches: Vec<LinearAttnCache>,
}

/// SSD-tier hydrate source: owns an [`SsdKvIndex`] + the model identity.
///
/// One per loaded model (its `model_id` doubles as the spill namespace, exactly
/// as [`super::spill::SsdSpiller`]). The index is opened once at construction
/// and reused for every lookup.
#[allow(missing_debug_implementations)]
pub struct SsdHydrator {
    index: SsdKvIndex,
    // Resolved namespace dir, kept for parity with `SsdSpiller` + test
    // construction; production `lookup` reads the absolute `row.path` from the
    // index, so this is not read on the hot path.
    #[allow(dead_code)]
    dir: PathBuf,
    model_id: String,
    kv_quant: KvQuant,
    /// stable u64 hash over the arch + KV layout for this snapshot.
    /// Salts the chained-hash digest stream and pins index lookups under the
    /// composite `(hash, layout_key)` PK.
    layout_key: u64,
    device: Device,
}

impl SsdHydrator {
    /// Open the hydrate source for `model_id` at the configured KV quant.
    ///
    /// The `.kvb` directory + index DB are resolved via
    /// `paths::kv_cache_dir(model_id)` — the same namespace the spiller
    /// writes to. Returns `Err` only if the index cannot be opened; the caller
    /// (model-load) decides whether to proceed without an SSD tier.
    pub fn open(
        model_id: impl Into<String>,
        kv_quant: KvQuant,
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
            kv_quant,
            layout_key,
            device,
        })
    }

    /// Test-only: build a hydrator bound to an explicit dir + already-open
    /// index, bypassing `paths::kv_cache_dir` so tests are hermetic.
    #[cfg(test)]
    pub fn with_index(
        model_id: impl Into<String>,
        kv_quant: KvQuant,
        layout_key: u64,
        device: Device,
        dir: PathBuf,
        index: SsdKvIndex,
    ) -> Self {
        Self {
            index,
            dir,
            model_id: model_id.into(),
            kv_quant,
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

    /// The KV quant this hydrator verifies + tags reconstructed entries with.
    pub fn kv_quant(&self) -> KvQuant {
        self.kv_quant
    }

    /// layout key in effect for this hydrator. Salts chained-hash
    /// digests and pins the `(hash, layout_key)` composite-PK lookup.
    pub fn layout_key(&self) -> u64 {
        self.layout_key
    }

    /// Index lookup + recompute of the RAM-side seeded block-hash chain for the
    /// matched block-aligned prefix. The seed partitions prompt-cache keys by KV
    /// layout and codec; the arch-side recompute after hydrate MUST use this exact
    /// formula or hydrated entries are unfindable in the RAM cache. (The index
    /// probe inside `lookup()` has its own seed — they must agree.)
    pub fn lookup_seeded(
        &self,
        prompt_ids: &[u32],
    ) -> rmlx_core::error::Result<Option<(HydratedBlock, Vec<u64>)>> {
        let Some(block) = self.lookup(prompt_ids)? else {
            return Ok(None);
        };
        let seed = FNV_OFFSET ^ self.layout_key ^ self.kv_quant.cache_key_salt();
        let hashes = chained_block_hashes_seeded(&block.prompt_ids, seed);
        Ok(Some((block, hashes)))
    }

    /// Look up + reconstruct the longest cached block-aligned prefix of
    /// `prompt_ids`. Returns `Ok(Some(_))` on an SSD hit, `Ok(None)` on a true
    /// miss **or** on corruption (after deleting the bad file + row + `warn!`).
    ///
    /// Never panics. The arch `SsdHydrate<E>` impl calls this and wraps the
    /// result as its concrete entry. Emits a [`SsdHydrateEvent`] via the
    /// process-global event recorder (if set) with per-phase timing.
    pub fn lookup(&self, prompt_ids: &[u32]) -> rmlx_core::error::Result<Option<HydratedBlock>> {
        self.lookup_inner(prompt_ids, None)
    }

    /// Test-only: like [`lookup`] but uses an explicit `EventRecorder` for
    /// hermetic event-capture tests, bypassing the process-global OnceLock.
    #[cfg(test)]
    pub fn lookup_with_recorder(
        &self,
        prompt_ids: &[u32],
        recorder: &EventRecorder,
    ) -> rmlx_core::error::Result<Option<HydratedBlock>> {
        self.lookup_inner(prompt_ids, Some(recorder))
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
    fn lookup_inner(
        &self,
        prompt_ids: &[u32],
        test_recorder: Option<&EventRecorder>,
    ) -> rmlx_core::error::Result<Option<HydratedBlock>> {
        // seed the chained walk with `FNV_OFFSET ^ layout_key ^
        // kv_quant.cache_key_salt()` so the probe digest stream is byte-identical
        // to the spill side's key (the spill/RAM-push/post-hydrate-recompute
        // paths all seed with this same salted value). Without the codec salt the
        // probe could never match a row the spill side wrote, so the SSD tier
        // would silently 0-hit. A zero `layout_key` with the codec salt is the
        // per-codec partition (same tokens under different codecs occupy disjoint
        // digest streams).
        let chained = chained_block_hashes_seeded(
            prompt_ids,
            FNV_OFFSET ^ self.layout_key ^ self.kv_quant.cache_key_salt(),
        );
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
        match read_caches_timed(&row.path, self.device, &self.model_id, self.kv_quant) {
            Ok((
                kv_caches,
                lin_caches,
                bytes_read,
                dur_read_us,
                dur_dequant_us,
                dur_finalize_us,
            )) => {
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

                let prefix_len = block_count * BLOCK_TOKENS;
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
                    prompt_ids: prompt_ids[..prefix_len].to_vec(),
                    kv_caches,
                    lin_caches,
                }))
            }
            Err(e) => {
                // Corrupt / metadata-mismatch / read error: delete file + row,
                // fall through to prefill.
                tracing::warn!(
                    hash = %row.hash,
                    path = %row.path.display(),
                    error = %e,
                    "ssd-hydrate: corrupt block, deleting file + index row, falling back to prefill"
                );
                let _ = std::fs::remove_file(&row.path);
                if let Err(de) = self.index.delete(&row.hash, row.layout_key) {
                    tracing::warn!(
                        hash = %row.hash,
                        layout_key = row.layout_key,
                        error = %de,
                        "ssd-hydrate: index row delete failed"
                    );
                }
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
#[path = "hydrate_tests.rs"]
mod tests;
