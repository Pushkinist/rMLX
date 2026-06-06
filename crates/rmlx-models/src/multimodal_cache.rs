// unsafe_code: pixel_f32_bytes / pcm_f32_bytes reinterpret f32 buffers as
// byte slices for hashing. Read-only, lifetime-bound, no aliasing.
#![allow(unsafe_code)]

//! Multimodal encoder-output cache.
//!
//! Caches the post-encoder `Array` produced by a vision tower (or audio
//! encoder) keyed on a content hash of the *post-preprocess* tensor bytes.
//! On a cache hit, the caller skips the encoder forward entirely and reuses
//! the cached embedding array.
//!
//! ## Design
//!
//! - **Single shared instance** per `AppState`, used by every vision/audio
//!   modality. Wrap in `Arc<MultimodalCache>` for cheap clone.
//! - **Hash recipe** (lifted from
//!   `dynamo/components/.../vllm/multimodal_utils/hash_utils.py:32-95`):
//!   fixed 12-byte header `[version:u8=1, mode:u8 (0=image | 1=audio),
//!   dtype:u8 (0=bf16 | 1=f32), channels:u8, dim1:u16 LE, dim2:u16 LE,
//!   reserved:u32 LE]` + the canonical preprocessed pixel/PCM bytes.
//!   The fixed header prevents `(H,W)` collision when the same pixel byte
//!   stream is reshaped to a different geometry. For audio, `dim1`/`dim2` are
//!   `0` and the `reserved` field carries the sample rate; `channels` lands
//!   in `header[3]` so mono vs stereo PCM byte runs cannot alias; `n_samples`
//!   is implied by the trailing byte length.
//! - **Hasher**: `twox-hash` xxh3_64 (MIT, tiny, no transitive cost). Stored
//!   as 8 bytes (the raw digest). If the digest is ever widened, bump the
//!   `version` byte in the header along with the key array size — this type
//!   is internal-only, no external ABI depends on the byte layout.
//! - **Eviction**: byte-budget LRU. Apple Silicon shared memory makes the
//!   embedding tensor count uninformative — bytes are honest. When `put`
//!   would push `used_bytes` over `budget_bytes`, the entry with the smallest
//!   `last_used` tick is dropped until the new entry fits. A `put` whose own
//!   `byte_size` exceeds the budget short-circuits to a no-op (no half-fills).
//! - **Thread-safety**: a single inner `Mutex`. Vision forward today is
//!   single-threaded — concurrent calls only occur when more than one decode
//!   thread is wired through the same vision tower (not the case in `0.1.0`,
//!   but the FIFO admission queue and `audio` path can interleave). The
//!   mutex covers the whole `get`/`put` critical section; the encoder
//!   forward runs **outside** the mutex so the cache never holds the GPU.
//!
//! ## Hash-collision safety
//!
//! xxh3_64 is non-cryptographic. For trusted internal keyspaces (this is
//! decoder-internal — neither HTTP request bodies nor JSON values reach
//! this hasher), the practical collision rate over a process lifetime is
//! negligible. Should a single bit-flip of the cached embedding ever ship
//! to a user that would normally re-encode, the failure mode is "model
//! produces wrong-looking output for one request" (no correctness invariant
//! is violated downstream — the embedding is the model's own opaque vector).
//! If that ever becomes a real concern, swap the digest computation to a
//! cryptographic hash without changing the type.
//!
//! ## What this is not
//!
//! - It is not a **HTTP-URL** cache. URL re-fetch dedup is explicitly
//!   out of scope here.
//! - It is not the **prompt KV-cache**. Encoder embeddings are pre-LM, the
//!   KV cache is post-LM. The two caches share no entries.

use std::collections::HashMap;
use std::hash::Hasher as _;
use std::mem::size_of_val;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use rmlx_mlx::Array;
use tracing::Level;
use twox_hash::xxhash3_64::Hasher as XxHash3_64Hasher;

/// Mode discriminator embedded in the 12-byte header.
#[allow(
    clippy::exhaustive_enums,
    reason = "fixed wire-format byte: adding a variant is a breaking change to the digest layout and requires bumping `version` in the header"
)]
#[allow(missing_docs)]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MmMode {
    Image = 0,
    Audio = 1,
}

/// Dtype discriminator embedded in the 12-byte header. Kept independent of
/// the MLX `Dtype` enum so the on-the-wire byte never breaks when MLX adds a
/// new element type.
#[allow(
    clippy::exhaustive_enums,
    reason = "fixed wire-format byte: adding a variant is a breaking change to the digest layout and requires bumping `version` in the header"
)]
#[allow(missing_docs)]
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MmDtype {
    Bf16 = 0,
    F32 = 1,
}

/// Content-hash key for the cache. 8 bytes (xxh3_64 digest).
///
/// If the digest is ever widened (e.g. blake3 256-bit), bump the `version`
/// byte in the 12-byte header and grow this array — the on-the-wire layout
/// is internal-only, no external ABI depends on the size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MmCacheKey([u8; 8]);

impl MmCacheKey {
    /// Hex-encoded short fingerprint for log messages (first 8 hex chars =
    /// first 4 bytes of the digest). NEVER use as a security identifier.
    #[must_use]
    pub fn short_hex(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::with_capacity(8);
        for b in &self.0[..4] {
            let _ = write!(out, "{b:02x}");
        }
        out
    }

    /// Build an image key. `pixel_bytes` is the **post-preprocess** flat
    /// pixel buffer (e.g. the same `&[f32]` the vision tower reads). The
    /// 12-byte header prevents `(H,W)` reshape collisions.
    #[must_use]
    pub fn image_key(
        pixel_bytes: &[u8],
        height: u16,
        width: u16,
        channels: u8,
        dtype: MmDtype,
    ) -> Self {
        let header = build_header(MmMode::Image, dtype, channels, height, width, 0);
        Self(digest(&header, pixel_bytes))
    }

    /// Build an audio key. `pcm_bytes` is the **post-preprocess** PCM byte
    /// stream (typically f32 mono). `sample_rate` lands in the `reserved`
    /// field so two identical PCM byte runs at different sample rates do not
    /// alias. `dim1`/`dim2` are unused for audio (set to 0).
    #[must_use]
    pub fn audio_key(pcm_bytes: &[u8], sample_rate: u32, dtype: MmDtype, channels: u8) -> Self {
        let header = build_header(MmMode::Audio, dtype, channels, 0, 0, sample_rate);
        Self(digest(&header, pcm_bytes))
    }

    /// Raw 8-byte digest. Public for tests + future on-disk serialization.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }
}

fn build_header(
    mode: MmMode,
    dtype: MmDtype,
    channels: u8,
    dim1: u16,
    dim2: u16,
    reserved: u32,
) -> [u8; 12] {
    let mut header = [0u8; 12];
    header[0] = 1; // version
    header[1] = mode as u8;
    header[2] = dtype as u8;
    header[3] = channels;
    header[4..6].copy_from_slice(&dim1.to_le_bytes());
    header[6..8].copy_from_slice(&dim2.to_le_bytes());
    header[8..12].copy_from_slice(&reserved.to_le_bytes());
    header
}

fn digest(header: &[u8; 12], payload: &[u8]) -> [u8; 8] {
    // Streaming xxh3_64 over (header || payload). Avoids the ~9.6 MiB temp
    // allocation a one-shot path would need for a 896×896×3 f32 image.
    // `alloc` feature on `twox-hash` is enabled at the workspace level for
    // the `xxhash3_64::Hasher::new` constructor.
    let mut h = XxHash3_64Hasher::new();
    h.write(header);
    h.write(payload);
    h.finish().to_le_bytes()
}

/// Compute the in-use byte size of an `Array`. Equals
/// `shape.product() * dtype.itemsize()`. The caller passes this into
/// [`MultimodalCache::put`] so the cache does not need to hold a live
/// reference to the array to compute its footprint.
///
/// Returns `Error::Model` on a negative shape dimension; MLX should never
/// surface such an array, but the previous silent `.max(0)` defense could
/// have cached a multi-MiB entry as 0 bytes, corrupting the byte budget.
pub fn array_byte_size(a: &Array) -> rmlx_core::error::Result<usize> {
    let mut elems: usize = 1;
    for d in a.shape() {
        if d < 0 {
            return Err(rmlx_core::error::Error::Model(format!(
                "multimodal_cache: array_byte_size received negative dim {d} in shape {:?}",
                a.shape()
            )));
        }
        let du = usize::try_from(d).map_err(|e| {
            rmlx_core::error::Error::Model(format!(
                "multimodal_cache: dim {d} does not fit usize: {e}"
            ))
        })?;
        elems = elems.saturating_mul(du);
    }
    Ok(elems.saturating_mul(a.dtype().itemsize()))
}

/// One cached entry: the encoder output array(s) + bookkeeping.
///
/// `arrays` is a small `Vec` so a single entry can hold a multi-array
/// payload (e.g. Qwen3-VL-MoE returns the merged image embeds + per-layer
/// deepstack embeds from a single ViT pass — caching them together skips
/// the whole vision tower).
#[allow(missing_debug_implementations)]
struct MmCacheEntry {
    arrays: Vec<Array>,
    byte_size: usize,
    last_used: u64,
}

/// Snapshot of cache counters. Returned by [`MultimodalCache::stats`].
#[allow(
    clippy::exhaustive_structs,
    reason = "snapshot DTO — fields are read-only for callers"
)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MmCacheStats {
    /// Number of `get` calls that returned `Some`.
    pub hits: u64,
    /// Number of `get` calls that returned `None`.
    pub misses: u64,
    /// Live bytes currently held by the cache (sum of `byte_size` over all entries).
    pub used_bytes: usize,
    /// Configured byte budget. `0` means the cache is disabled.
    pub capacity_bytes: usize,
    /// Live entry count.
    pub entries: usize,
    /// Cumulative number of metrics-event emit failures from the SQLite
    /// recorder. Non-zero implies the events stream may be incomplete.
    pub recorder_errors: u64,
}

/// Byte-budget LRU cache for encoder-output arrays.
pub struct MultimodalCache {
    inner: Mutex<Inner>,
    budget_bytes: usize,
    tick: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    /// Cumulative count of SQLite recorder emit failures. Surfaced via
    /// [`MmCacheStats::recorder_errors`].
    recorder_errors: AtomicU64,
    /// Optional event sink. When set (via [`MultimodalCache::set_recorder`]),
    /// every `get`/`put` emits a `mm_cache_hit` / `mm_cache_miss` / `mm_cache_insert`
    /// event into the metrics DB. Default `None` keeps unit-test paths zero-cost.
    recorder: std::sync::RwLock<Option<Arc<rmlx_metrics::events::EventRecorder>>>,
    /// Optional model identifier emitted as `model_path` on each event. Set
    /// alongside the recorder; defaults to "(unknown)". `Arc<str>` so read
    /// path is a refcount bump, not a `String::clone`.
    model_tag: std::sync::RwLock<Arc<str>>,
}

impl std::fmt::Debug for MultimodalCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.stats();
        f.debug_struct("MultimodalCache")
            .field("capacity_bytes", &s.capacity_bytes)
            .field("used_bytes", &s.used_bytes)
            .field("entries", &s.entries)
            .field("hits", &s.hits)
            .field("misses", &s.misses)
            .finish()
    }
}

struct Inner {
    map: HashMap<MmCacheKey, MmCacheEntry>,
    used_bytes: usize,
}

impl MultimodalCache {
    /// Create a new cache with the given byte budget. `budget_bytes == 0`
    /// disables the cache — `get` always returns `None`, `put` is a no-op.
    /// All public methods remain valid (callers do not have to special-case
    /// the disabled state).
    #[must_use]
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                used_bytes: 0,
            }),
            budget_bytes,
            tick: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            recorder_errors: AtomicU64::new(0),
            recorder: std::sync::RwLock::new(None),
            model_tag: std::sync::RwLock::new(Arc::<str>::from("(unknown)")),
        }
    }

    /// Install (or replace) the metrics-event recorder. After this call
    /// every `get`/`put` emits a `mm_cache_hit` / `mm_cache_miss` /
    /// `mm_cache_insert` row into the `events` SQLite table.
    ///
    /// `model_tag` is the value written into the row's `model_path` column;
    /// pass a stable identifier (e.g. the loaded model id, or `"global"` if
    /// the cache is shared across models).
    pub fn set_recorder(
        &self,
        recorder: Arc<rmlx_metrics::events::EventRecorder>,
        model_tag: impl Into<Arc<str>>,
    ) {
        if let Ok(mut g) = self.recorder.write() {
            *g = Some(recorder);
        } else {
            tracing::error!("mm_cache: recorder rwlock poisoned; events disabled");
        }
        if let Ok(mut g) = self.model_tag.write() {
            *g = model_tag.into();
        } else {
            tracing::error!("mm_cache: model_tag rwlock poisoned; tag not updated");
        }
    }

    fn emit_event(&self, op: &str, value: f64, notes: &str) {
        // Gate `mm_cache_miss` behind debug-level: misses are the dominant
        // cold-path event and carry no actionable signal at info level.
        // Hits and inserts always emit (cache effectiveness gauge).
        if op == "mm_cache_miss" && !tracing::enabled!(Level::DEBUG) {
            return;
        }
        let Ok(rec_guard) = self.recorder.read() else {
            tracing::error!("mm_cache: recorder rwlock poisoned on read; events disabled");
            return;
        };
        let Some(rec) = rec_guard.as_ref() else {
            return;
        };
        let tag: Arc<str> = if let Ok(g) = self.model_tag.read() {
            Arc::clone(&g)
        } else {
            tracing::error!("mm_cache: model_tag rwlock poisoned on read; using fallback");
            Arc::<str>::from("(unknown)")
        };
        if let Err(e) = rec.record(&rmlx_metrics::events::Measurement {
            model_path: &tag,
            quant_mode: "",
            stage: "mm_cache",
            op,
            value_unit: "count",
            value,
            notes,
        }) {
            self.recorder_errors.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(error = %e, op, "mm_cache: event emit failed");
        }
    }

    /// `true` when the configured budget is zero (cache disabled).
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.budget_bytes == 0
    }

    /// Look up `key`. On hit, returns a clone of the cached single `Array`
    /// (mlx-c reference-counted handles, no data copy) and bumps the LRU
    /// recency. Returns `None` if the entry holds more than one array — use
    /// [`get_many`](Self::get_many) for multi-array payloads.
    ///
    /// Primitive single-array path: avoids the `Vec` allocation that wrapping
    /// `get_many` would incur on the dominant hit path.
    pub fn get(&self, key: &MmCacheKey) -> Option<Array> {
        if self.budget_bytes == 0 {
            return None;
        }
        let Ok(mut inner) = self.inner.lock() else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("mm_cache: mutex poisoned on get; treating as miss");
            return None;
        };
        let Some(entry) = inner.map.get_mut(key) else {
            drop(inner);
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.emit_event("mm_cache_miss", 1.0, "");
            return None;
        };
        // Multi-array entry: caller must use `get_many`. Treat as miss
        // without bumping LRU.
        if entry.arrays.len() != 1 {
            drop(inner);
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.emit_event("mm_cache_miss", 1.0, "");
            return None;
        }
        // L4: clone first; only bump `last_used` + `hits` on success.
        let Some(first) = entry.arrays.first() else {
            drop(inner);
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.emit_event("mm_cache_miss", 1.0, "");
            return None;
        };
        let Ok(cloned) = first.try_clone() else {
            drop(inner);
            tracing::warn!("mm_cache: array try_clone failed; treating as miss");
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let bytes = entry.byte_size;
        // L6: only bump tick on a confirmed hit.
        let now = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        entry.last_used = now;
        self.hits.fetch_add(1, Ordering::Relaxed);
        drop(inner);
        tracing::debug!(key = %key.short_hex(), bytes, arrays = 1, "mm_cache: hit");
        self.emit_event("mm_cache_hit", 1.0, &format!("bytes={bytes}"));
        Some(cloned)
    }

    /// Look up `key`. On hit, returns clones of all cached arrays for this
    /// key in insertion order and bumps the LRU recency.
    pub fn get_many(&self, key: &MmCacheKey) -> Option<Vec<Array>> {
        if self.budget_bytes == 0 {
            return None;
        }
        let Ok(mut inner) = self.inner.lock() else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            tracing::warn!("mm_cache: mutex poisoned on get_many; treating as miss");
            return None;
        };
        let Some(entry) = inner.map.get_mut(key) else {
            drop(inner);
            self.misses.fetch_add(1, Ordering::Relaxed);
            self.emit_event("mm_cache_miss", 1.0, "");
            return None;
        };
        // L4: read every clone into a scratch Vec first; only on full success
        // do we touch `last_used` / `hits`. A partial-clone failure must not
        // distort LRU.
        let mut out = Vec::with_capacity(entry.arrays.len());
        for a in &entry.arrays {
            match a.try_clone() {
                Ok(c) => out.push(c),
                Err(e) => {
                    drop(inner);
                    tracing::warn!(error = %e, "mm_cache: array try_clone failed; treating as miss");
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
        }
        let bytes = entry.byte_size;
        let arrays_len = entry.arrays.len();
        // L6: only bump tick on a confirmed hit.
        let now = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        entry.last_used = now;
        self.hits.fetch_add(1, Ordering::Relaxed);
        drop(inner);
        tracing::debug!(
            key = %key.short_hex(),
            bytes,
            arrays = arrays_len,
            "mm_cache: hit"
        );
        self.emit_event("mm_cache_hit", 1.0, &format!("bytes={bytes}"));
        Some(out)
    }

    /// Insert a single `array` under `key`. Primitive single-array path:
    /// avoids the `vec![array]` allocation a `put_many` wrapper would incur.
    /// See [`put_many`](Self::put_many) for the multi-array variant.
    pub fn put(&self, key: MmCacheKey, array: Array, byte_size: usize) {
        // We still need owned `Vec` storage in the entry, but the dominant
        // single-array hit path runs through `get` (no Vec), and inserts are
        // off the critical loop.
        self.put_inner(key, vec![array], byte_size);
    }

    /// Insert one or more `arrays` under `key`. If `byte_size > budget_bytes`,
    /// the call is a no-op (the entry would force eviction of itself).
    /// Existing entries with the same key are overwritten and their old
    /// `byte_size` is reclaimed.
    pub fn put_many(&self, key: MmCacheKey, arrays: Vec<Array>, byte_size: usize) {
        self.put_inner(key, arrays, byte_size);
    }

    #[allow(
        clippy::cognitive_complexity,
        reason = "sequential insert path: budget short-circuit + reclaim + eviction loop + insert + tracing/metrics emit — splitting into helpers would obscure the lock-held critical section"
    )]
    fn put_inner(&self, key: MmCacheKey, arrays: Vec<Array>, byte_size: usize) {
        if self.budget_bytes == 0 {
            return;
        }
        if byte_size > self.budget_bytes {
            tracing::debug!(
                key = %key.short_hex(),
                bytes = byte_size,
                budget = self.budget_bytes,
                "mm_cache: entry larger than budget; not inserted"
            );
            return;
        }
        let now = self.tick.fetch_add(1, Ordering::Relaxed) + 1;
        let Ok(mut inner) = self.inner.lock() else {
            tracing::warn!("mm_cache: mutex poisoned on put; dropping entry");
            return;
        };
        // Reclaim space held by the previous entry under this key, if any.
        if let Some(old) = inner.map.remove(&key) {
            inner.used_bytes = inner.used_bytes.saturating_sub(old.byte_size);
        }
        // Evict LRU until the new entry fits.
        while inner.used_bytes.saturating_add(byte_size) > self.budget_bytes {
            let Some(victim_key) = lru_key(&inner.map) else {
                break;
            };
            if let Some(victim) = inner.map.remove(&victim_key) {
                inner.used_bytes = inner.used_bytes.saturating_sub(victim.byte_size);
                tracing::debug!(
                    key = %victim_key.short_hex(),
                    bytes = victim.byte_size,
                    "mm_cache: evict (lru)"
                );
            } else {
                break;
            }
        }
        inner.used_bytes = inner.used_bytes.saturating_add(byte_size);
        inner.map.insert(
            key,
            MmCacheEntry {
                arrays,
                byte_size,
                last_used: now,
            },
        );
        let used = inner.used_bytes;
        let entries = inner.map.len();
        drop(inner);
        tracing::debug!(
            key = %key.short_hex(),
            bytes = byte_size,
            used,
            entries,
            "mm_cache: insert"
        );
        self.emit_event("mm_cache_insert", 1.0, &format!("bytes={byte_size}"));
    }

    /// Snapshot of the cache counters. Cheap (loads atomics + one lock for
    /// `used_bytes`/`entries`).
    pub fn stats(&self) -> MmCacheStats {
        let (used_bytes, entries) = match self.inner.lock() {
            Ok(g) => (g.used_bytes, g.map.len()),
            Err(_) => (0, 0),
        };
        MmCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            used_bytes,
            capacity_bytes: self.budget_bytes,
            entries,
            recorder_errors: self.recorder_errors.load(Ordering::Relaxed),
        }
    }

    /// Drop every entry. Stats counters are NOT reset (cumulative).
    pub fn clear(&self) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        inner.map.clear();
        inner.used_bytes = 0;
    }
}

/// Single-array get-or-compute helper hoisting the get/miss/insert
/// dance shared by every vision tower call site (gemma3, gemma4, jina_v4).
///
/// On a hit: returns the cached `Array` without invoking `compute`.
/// On a miss: runs `compute`, computes its byte size, attempts to clone it
/// into the cache (best-effort — clone failures are warned but do not fail
/// the caller), and returns the freshly computed array.
///
/// `cache == None` short-circuits straight to `compute()`.
pub fn get_or_compute<F>(
    cache: Option<&MultimodalCache>,
    key: MmCacheKey,
    compute: F,
) -> rmlx_core::error::Result<Array>
where
    F: FnOnce() -> rmlx_core::error::Result<Array>,
{
    let Some(cache) = cache else {
        return compute();
    };
    if let Some(cached) = cache.get(&key) {
        return Ok(cached);
    }
    let computed = compute()?;
    match array_byte_size(&computed) {
        Ok(sz) => {
            if let Ok(clone) = computed.try_clone() {
                cache.put(key, clone, sz);
            } else {
                tracing::warn!(key = %key.short_hex(), "mm_cache: try_clone on insert failed; not caching");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, key = %key.short_hex(), "mm_cache: array_byte_size failed; not caching");
        }
    }
    Ok(computed)
}

// `lru_key` is a linear scan over `map`; `put_many` calls it once per
// eviction step, giving an O(n^2) worst case if `n` entries must be evicted
// in one `put_many`. At the default 512 MiB budget and typical encoder-output
// sizes (a 896x896x3 Gemma image embeds at ~1–4 MiB), `n` is bounded at
// roughly 100–500 entries; eviction bursts are amortized across many `put`s
// at steady state and the scan stays well under a microsecond.
// FOLLOWUP: if the budget grows to multi-GiB or workloads emerge that
// churn the cache (many small audio clips per request), upgrade to a
// `BinaryHeap<(last_used, key)>` or a hand-rolled LRU list. No external dep
// required and we keep the "ask before adding a dep" rule.
fn lru_key(map: &HashMap<MmCacheKey, MmCacheEntry>) -> Option<MmCacheKey> {
    map.iter().min_by_key(|(_, e)| e.last_used).map(|(k, _)| *k)
}

/// Reinterpret a host `&[f32]` buffer as raw bytes. Used by vision-tower
/// callers to hash pixel data with a single pass — copying into an
/// intermediate `Vec<u8>` would double the per-encode memory traffic for
/// large images. The lifetime of the returned slice is tied to the input.
#[must_use]
pub fn pixel_f32_bytes(pixels: &[f32]) -> &[u8] {
    let len_bytes = size_of_val(pixels);
    // SAFETY: f32 is `Pod` (4 bytes, no padding), the slice is read-only,
    // and the returned reference has the same lifetime as `pixels`.
    unsafe { std::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), len_bytes) }
}

/// Reinterpret a host PCM `&[f32]` waveform buffer as raw bytes.
#[must_use]
pub fn pcm_f32_bytes(pcm: &[f32]) -> &[u8] {
    pixel_f32_bytes(pcm)
}

#[cfg(test)]
#[path = "multimodal_cache_tests.rs"]
mod multimodal_cache_tests;
