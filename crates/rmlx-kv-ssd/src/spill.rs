//! Background SSD-spill executor for RAM-evicted prompt-cache entries.
//!
//! When `PromptCache::push` evicts an entry (RAM-cap or slot-count), the entry
//! is offered to a [`crate::prompt_cache::SpillSink`] before being dropped.
//! The production sink builds an [`SpillJob`] — a cheap refcount-clone of the
//! evicted entry's caches plus its identity metadata — and `try_send`s it onto
//! a **bounded `std::sync::mpsc::sync_channel`**. A single dedicated drain
//! thread receives jobs, serializes them via 's `block_io::write_caches`
//! to a `.kvb` file under `paths::kv_cache_dir(namespace)`, and records the
//! file in 's `SsdKvIndex`.
//!
//! ## Why a channel + drain thread (not `spawn_blocking`)
//!
//! `PromptCache` is sync, called from the sync inference path. Threading a
//! Tokio handle into the cache layer would leak async into compute. A bounded
//! `sync_channel` + one drain thread keeps the hot path sync and **never
//! blocks on disk I/O**: the hot path only does a refcount-clone of the caches
//! (no tensor copy) and a non-blocking `try_send`. If the channel is full the
//! job is `warn!`-dropped — back-pressure never stalls decode.
//!
//! ## Where the host-materialization happens
//!
//! Serialization (forcing the MLX arrays to the host via `to_bytes()`) is the
//! expensive part and runs **in the drain thread**, inside
//! `block_io::write_caches`. The hot path moves only the (refcount-cloned)
//! caches into the job. This matches the decision to keep that work off
//! the inference thread.
//!
//! ## Failure containment
//!
//! Every drain-thread step is fire-and-forget: a serialize or index error is
//! `warn!`ed with context and the job is dropped. The drain thread never
//! panics; the spill is a pure side effect of eviction and never affects what
//! `push`/lookup return or the in-RAM cache semantics.
//!
//! The public surface (`SsdSpiller::spawn`, `try_spill`, `SpillJob`, and
//! `PromptCache::set_spill_sink`) is wired into production by :
//! `ssd_tier::attach_at_load` spawns a spiller per loaded model and
//! `set_spill_sink`s it onto the per-arch `PROMPT_CACHE`, gated by
//! `--kv-ssd-cache-gb`.

use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::thread;
use std::time::Instant;

use rmlx_metrics::events::{EventRecorder, SsdSpillEvent};
use rmlx_mlx::Device;

use rmlx_kv_quant::kvcache::KvCache;
use rmlx_kv_quant::linear_attn::LinearAttnCache;
use rmlx_kv_quant::KvQuant;

use crate::block_io::write_caches_timed;
use crate::hooks::{call_ssd_spill_prom_hook, ssd_event_recorder};
use crate::ssd_index::{hash_to_hex, SsdKvIndex};

/// Bounded spill channel depth. Small on purpose: spill is best-effort, and a
/// backlog means the disk cannot keep up — dropping is the correct behavior.
const SPILL_CHANNEL_CAP: usize = 16;

/// A unit of spill work: one evicted entry's caches + identity metadata.
///
/// Built on the hot path (cheap refcount-clone of the caches), serialized in
/// the drain thread.
#[allow(missing_debug_implementations)]
#[allow(
    clippy::exhaustive_structs,
    reason = "promoted: internal SSD-tier bridge struct, was pub(crate); promoted to pub for cross-crate use by arch SpillSink impls in rmlx-models; adding a field is a coordinated update across both crates"
)]
pub struct SpillJob {
    /// Spill key — the last chained-block digest of the entry's prompt under
    /// the active `layout_key` salt. Hex-formatted for the `.kvb`
    /// filename stem and the index `hash` column.
    pub hash: u64,
    /// layout key for this spill — stable u64 over the arch + KV layout.
    /// Threaded into the `kv_blocks` row's `layout_key` column so spilled
    /// blocks belong only to their own layout.
    pub layout_key: u64,
    /// `<arch>/<snapshot>` identity, written to the block header + index row.
    pub model_id: String,
    /// KV quant in effect for this snapshot (block header + index row).
    pub kv_quant: KvQuant,
    /// Attention KV caches (refcount-cloned from the evicted entry).
    pub kv_caches: Vec<KvCache>,
    /// GDN linear-attention recurrent state (empty for pure-attention archs).
    pub lin_caches: Vec<LinearAttnCache>,
}

/// Fire-and-forget SSD spiller: a bounded sender + a background drain thread.
///
/// The drain thread owns the [`SsdKvIndex`] and the destination directory, and
/// lives for the process lifetime (the join handle is dropped — the thread
/// exits when all senders are dropped).
///
/// It is also where the namespace's on-disk budget is enforced. The drain
/// thread is the only writer that grows the tier, it already holds an index
/// handle, and it runs off the inference path — so it evicts to budget after
/// every block it records. Nothing else evicts between model loads, so without
/// that pass a long-lived `serve` would exceed `--kv-ssd-cache-gb` for its
/// whole lifetime and only come back under the ceiling at the next attach.
#[allow(missing_debug_implementations)]
pub struct SsdSpiller {
    tx: SyncSender<SpillJob>,
    model_id: String,
    /// stable u64 hash over the arch + KV layout. Stamped on every
    /// spilled row's `layout_key` column so the row belongs only to its own
    /// layout (composite `(hash, layout_key)` PK).
    layout_key: u64,
}

impl SsdSpiller {
    /// Spawn the drain thread for `model_id` and return a handle.
    ///
    /// `model_id` (`<arch>/<snapshot>`) doubles as the spill namespace: it
    /// selects the `paths::kv_cache_dir(model_id)` directory and the
    /// `SsdKvIndex` DB, and is written to each block header + index row. The
    /// per-entry `KvQuant` is recorded per-row, so one directory per model
    /// holds all quant variants. The index is opened **on the drain thread**
    /// so the hot path never touches SQLite. `device` is the device used for
    /// serialization.
    ///
    /// On index-open failure the drain thread logs `warn!` and exits; the
    /// sender side still accepts (and silently drops) jobs, so the cache keeps
    /// working with spill effectively disabled.
    ///
    /// The namespace byte ceiling is resolved once here from the installed
    /// SSD-tier config, so it is the same figure the attach-time maintenance
    /// pass evicted to. `0` (tier unconfigured) leaves the drain thread with no
    /// ceiling to enforce, exactly as before.
    pub fn spawn(model_id: impl Into<String>, layout_key: u64, device: Device) -> Self {
        let model_id = model_id.into();
        let (tx, rx) = sync_channel::<SpillJob>(SPILL_CHANNEL_CAP);

        let budget_bytes = crate::ssd_tier::active()
            .as_ref()
            .map_or(0, crate::ssd_tier::effective_namespace_budget);
        let ns = model_id.clone();
        let _ = thread::Builder::new()
            .name("rmlx-kv-spill".into())
            .spawn(move || {
                let dir = rmlx_core::paths::kv_cache_dir(&ns);
                let index = match SsdKvIndex::open(&ns) {
                    Ok(idx) => idx,
                    Err(e) => {
                        tracing::warn!(namespace = %ns, error = %e, "kv-spill: index open failed, spill disabled");
 // Drain and drop remaining jobs so senders don't block.
                        for _ in rx {}
                        return;
                    }
                };
                tracing::info!(namespace = %ns, dir = %dir.display(), budget_bytes, "kv-spill: drain thread started");
                for job in rx {
                    if drain_one(&index, &dir, device, job) {
                        crate::ssd_tier::enforce_namespace_budget(&index, &ns, budget_bytes);
                    }
                }
                tracing::debug!(namespace = %ns, "kv-spill: drain thread exiting (all senders dropped)");
            });

        Self {
            tx,
            model_id,
            layout_key,
        }
    }

    /// Test-only: spawn a drain thread bound to an explicit `dir` + `index`,
    /// bypassing `paths::kv_cache_dir` so tests are hermetic (no env-var
    /// coupling, no shared workspace `.rmlx/`). Same drain semantics as
    /// [`Self::spawn`], including the post-spill evict-to-budget pass —
    /// `budget_bytes` stands in for what [`Self::spawn`] reads from the
    /// installed config (`0` = no ceiling). Returns the handle plus the spawned
    /// thread's `JoinHandle` so a test can join after dropping the sender.
    #[cfg(test)]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    pub fn spawn_with_index(
        model_id: impl Into<String>,
        layout_key: u64,
        device: Device,
        dir: PathBuf,
        index: SsdKvIndex,
        budget_bytes: u64,
    ) -> (Self, thread::JoinHandle<()>) {
        let model_id = model_id.into();
        let ns = model_id.clone();
        let (tx, rx) = sync_channel::<SpillJob>(SPILL_CHANNEL_CAP);
        let handle = thread::Builder::new()
            .name("rmlx-kv-spill-test".into())
            .spawn(move || {
                for job in rx {
                    if drain_one_inner(&index, &dir, device, job, None) {
                        crate::ssd_tier::enforce_namespace_budget(&index, &ns, budget_bytes);
                    }
                }
            })
            .expect("spawn test drain thread");
        (
            Self {
                tx,
                model_id,
                layout_key,
            },
            handle,
        )
    }

    /// Test-only: same as [`spawn_with_index`] but also injects an explicit
    /// `EventRecorder` so tests can verify SSD-tier event rows in a hermetic
    /// temp DB, bypassing the process-global `SSD_EVENT_RECORDER` OnceLock.
    #[cfg(test)]
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    pub fn spawn_with_index_and_recorder(
        model_id: impl Into<String>,
        layout_key: u64,
        device: Device,
        dir: PathBuf,
        index: SsdKvIndex,
        recorder: std::sync::Arc<EventRecorder>,
    ) -> (Self, thread::JoinHandle<()>) {
        let model_id = model_id.into();
        let (tx, rx) = sync_channel::<SpillJob>(SPILL_CHANNEL_CAP);
        let handle = thread::Builder::new()
            .name("rmlx-kv-spill-test-rec".into())
            .spawn(move || {
                for job in rx {
                    drain_one_inner(&index, &dir, device, job, Some(&recorder));
                }
            })
            .expect("spawn test drain thread");
        // No budget pass: this variant exists to observe the emitted spill
        // event, and an eviction racing that observation would make the row
        // count it asserts on depend on timing.
        (
            Self {
                tx,
                model_id,
                layout_key,
            },
            handle,
        )
    }

    /// `<arch>/<snapshot>` identity this spiller persists under.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// layout key this spiller stamps onto every spilled row.
    pub fn layout_key(&self) -> u64 {
        self.layout_key
    }

    /// Enqueue a spill job without blocking. Drops + `warn!`s on a full or
    /// disconnected channel — never stalls the caller (the decode hot path).
    pub fn try_spill(&self, job: SpillJob) {
        let hash = job.hash;
        match self.tx.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                tracing::warn!(
                    hash = %hash_to_hex(hash),
                    "kv-spill: channel full, dropping spill job (disk can't keep up)"
                );
            }
            Err(TrySendError::Disconnected(_)) => {
                tracing::warn!(
                    hash = %hash_to_hex(hash),
                    "kv-spill: drain thread gone, dropping spill job"
                );
            }
        }
    }
}

/// Serialize one job to `<dir>/<hash>.kvb` and record it in the index.
///
/// Fire-and-forget: any error is `warn!`ed and the job dropped. Never panics.
/// Emits a [`SsdSpillEvent`] via the process-global event recorder (if set)
/// with per-phase timing and byte count.
///
/// Returns `true` only when the block reached the index, which is the one
/// outcome that grew the namespace and therefore the only one worth running an
/// evict-to-budget pass after.
fn drain_one(index: &SsdKvIndex, dir: &std::path::Path, device: Device, job: SpillJob) -> bool {
    drain_one_inner(index, dir, device, job, None)
}

/// Core of [`drain_one`], factored so tests can inject an explicit recorder.
///
/// Production `drain_one` passes `None` (uses the process-global OnceLock).
/// Tests pass `Some(&recorder)` to capture events in a hermetic temp DB.
#[allow(
    clippy::cognitive_complexity,
    reason = "serialize → index-record → event-emit → prometheus-hook: four \
              sequential best-effort phases each with their own timing and error \
              branches; splitting would scatter timing state across fn boundaries"
)]
fn drain_one_inner(
    index: &SsdKvIndex,
    dir: &std::path::Path,
    device: Device,
    job: SpillJob,
    test_recorder: Option<&EventRecorder>,
) -> bool {
    let SpillJob {
        hash,
        layout_key,
        model_id,
        kv_quant,
        kv_caches,
        lin_caches,
    } = job;
    let hex = hash_to_hex(hash);
    let path: PathBuf = dir.join(format!("{hex}.kvb"));

    // ── Phase 1+2: serialize (CPU eval) + write (FS) ─────────────────────────
    let t0 = Instant::now();
    let (byte_size, dur_serialize_us, dur_write_us) =
        match write_caches_timed(&path, device, &model_id, kv_quant, &kv_caches, &lin_caches) {
            Ok(timings) => timings,
            Err(e) => {
                tracing::warn!(
                    hash = %hex,
                    path = %path.display(),
                    error = %e,
                    "kv-spill: serialize failed, dropping job"
                );
                let _ = std::fs::remove_file(&path);
                return false;
            }
        };

    // ── Phase 3: index record ─────────────────────────────────────────────────
    let t_idx = Instant::now();
    if let Err(e) = index.record(
        &hex,
        layout_key,
        &path,
        &model_id,
        &kv_quant.to_string(),
        byte_size,
    ) {
        tracing::warn!(
            hash = %hex,
            layout_key,
            path = %path.display(),
            error = %e,
            "kv-spill: index record failed, dropping job"
        );
        return false;
    }
    let dur_index_us = t_idx.elapsed().as_micros() as u64;
    let dur_us = t0.elapsed().as_micros() as u64;

    tracing::debug!(
        hash = %hex,
        path = %path.display(),
        byte_size,
        dur_us,
        dur_serialize_us,
        dur_write_us,
        dur_index_us,
        "kv-spill: block written + indexed"
    );

    // ── Emit per-block event ──────────────────────────────────────────────────
    // Use the injected test recorder if present; else fall back to the global.
    let global_rec;
    let rec_ref: Option<&EventRecorder> = if test_recorder.is_some() {
        test_recorder
    } else {
        global_rec = ssd_event_recorder();
        global_rec.as_deref()
    };

    if let Some(rec) = rec_ref {
        let ev = SsdSpillEvent {
            namespace: model_id,
            bytes: byte_size,
            dur_us,
            dur_serialize_us,
            dur_write_us,
            dur_index_us,
        };
        if let Err(e) = rec.record_ssd_spill(&ev) {
            tracing::warn!(
                hash = %hex,
                error = %e,
                "kv-spill: event recorder failed (non-fatal)"
            );
        }
    }

    // Feed the Prometheus histogram accumulator (registered by rmlx-server at
    // startup via `set_ssd_spill_prom_hook`). No-op when unset.
    call_ssd_spill_prom_hook(dur_us, byte_size);
    true
}

#[cfg(test)]
#[path = "spill_tests.rs"]
mod tests;
