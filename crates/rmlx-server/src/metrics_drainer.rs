//! SPSC async metrics drainer — F6 / L18.
//!
//! Per-request metrics (kv_cache_bytes, ttft_ms, prompt_cache_hits/misses/bytes,
//! load_phases) are emitted from the decode hot-path via `DrainerHandle::try_emit`.
//! A single async consumer task batches events and writes them to SQLite via the
//! existing `rmlx_metrics::Recorder`, keeping the blocking SQLite write off the
//! decode thread.
//!
//! # Invariants
//!
//! - Producer: `try_emit` is non-blocking. On channel full, the event is
//!   dropped and `dropped` counter incremented.
//! - Consumer: single tokio task, flushes every 100 ms OR when batch >= 32,
//!   whichever comes first.
//! - No lock is held across any `await`.
//! - SQLite write happens in `spawn_blocking` so the async executor thread is
//!   never blocked by disk I/O.

#![allow(
    clippy::cognitive_complexity,
    clippy::ref_option,
    clippy::too_many_lines
)]
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rmlx_metrics::identity::RunIdentity;
use rmlx_metrics::ingest::{MetricEntry, PromptRef, RunRecord, RunRecordBuilder};
use rmlx_metrics::migrate;
use rmlx_metrics::recorder::Recorder;
use rmlx_metrics::schema;
use tokio::sync::mpsc;

// ── Public types ──────────────────────────────────────────────────────────────

/// One telemetry event from the decode path.
///
/// Every variant carries the minimum context needed for the recorder to build
/// a `RunRecord`. Fields common to all variants (model_id, kv_quant, ts_utc)
/// are at the top level.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed event struct — five fields are the complete metrics-drainer event envelope; adding a field requires updating all MetricEvent construction sites in the server hot path"
)]
#[derive(Debug, Clone)]
pub struct MetricEvent {
    /// Snapshot basename (e.g. `mlx-community__gemma-4-e2b-it-mxfp8`).
    pub model_id: String,
    /// KV-quant label used by this request (e.g. `"k8v8"`).
    pub kv_quant: String,
    /// ISO-8601 UTC timestamp of the event.
    pub ts_utc: String,
    /// Server-side max context length in effect for this request
    /// (= `effective_max_ctx`, the resolved `--max-ctx` cap). Stored as
    /// `ctx_max` in the `observations` table (cell-identity column).
    /// Must be > 0 to satisfy `RunRecord::validate()`.
    pub ctx_max: i64,
    /// The metric value being reported.
    pub kind: MetricKind,
}

/// Which metric this event carries.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum MetricKind {
    /// KV-cache memory footprint in bytes for this request.
    KvCacheBytes(u64),
    /// Time-to-first-token latency in milliseconds.
    TtftMs(u64),
    /// Number of prompt-cache prefix hits for this request.
    PromptCacheHits(u64),
    /// Number of prompt-cache prefix misses for this request.
    PromptCacheMisses(u64),
    /// Bytes of prompt-cache prefix data reused for this request.
    PromptCacheBytes(u64),
    /// Number of KV block-cache hits.
    BlockHits(u64),
    /// Number of KV block-cache misses.
    BlockMisses(u64),
    /// Number of partial KV block hits.
    PartialHits(u64),
    /// cross-request hot (in-memory LRU) prompt-cache hit count.
    ///
    /// Same value as `PromptCacheHits` (the `PromptCache<E>` LRU is the hot
    /// in-RAM tier), emitted under the registry name `prompt_cache_hot_cache_hits`
    /// so the cross-request reuse win is queryable independently. Cumulative
    /// per running server (the cache is a process-wide per-arch static).
    HotCacheHits(u64),
    /// cross-request hot prompt-cache LRU eviction count.
    ///
    /// `CacheStats::evictions` — slots dropped by the slot-count / RAM-bytes
    /// LRU policy. Emitted under `prompt_cache_hot_cache_evictions`.
    HotCacheEvictions(u64),
    /// cross-tier SSD-hydrate hit count.
    ///
    /// `CacheStats::ssd_hits` — RAM misses that were served from the on-disk
    /// `.kvb` tier (longest-prefix block read back + promoted into RAM).
    /// Emitted under `prompt_cache_ssd_hits`. Cumulative per running server.
    SsdHits(u64),
    /// Per-model-load phase timings in milliseconds.
    LoadPhases {
        /// Time spent memory-mapping weight files.
        mmap_ms: f64,
        /// Time spent dequantizing weights to GPU format.
        dequant_ms: f64,
        /// Time spent transferring weights to GPU resident memory.
        gpu_residency_ms: f64,
        /// Time until the first Metal kernel is ready to execute.
        first_kernel_ready_ms: f64,
        /// Total load time from start to first-kernel-ready.
        total_ms: f64,
    },
    /// Per-token inter-token latency aggregates (M30).
    ///
    /// Emitted once per request after all decode steps complete.
    /// One event carries all three aggregates so the drainer writes a single
    /// `RunRecord` with three `MetricEntry` rows.
    ItlStats {
        /// Median inter-token latency in milliseconds.
        p50_ms: f64,
        /// 95th-percentile inter-token latency in milliseconds.
        p95_ms: f64,
        /// Mean inter-token latency in milliseconds.
        mean_ms: f64,
        /// Number of decode steps measured.
        step_count: usize,
    },
    /// C5 Slice A: milliseconds this request spent in the FIFO admission
    /// queue waiting for the single-GPU permit. Emitted once per admitted
    /// request at permit-acquire.
    QueueWaitMs(u64),
    /// C5 Slice A: in-flight admitted-request count observed at admission
    /// (this request inclusive) — the queue-depth gauge. Emitted once per
    /// admitted request alongside `QueueWaitMs`.
    QueueDepth(u64),
    /// C7: Metal allocator high-water mark, in MB (integer-divided from bytes).
    ///
    /// Read from `mlx_get_peak_memory` once per request at the same boundary
    /// as `KvCacheBytes`. Fulfils the `metal_peak_alloc_mb` registry Todo.
    MetalPeakAllocMb(u64),
    /// F1b: number of prompt (input) tokens for this request.
    ///
    /// Sourced from the same counter that populates the `Usage` response body.
    /// Emitted once per completed request from the handler (both blocking and
    /// streaming paths), never from the engine decode loop.
    PromptTokens(u32),
    /// F1b: number of completion (output/decode) tokens for this request.
    ///
    /// Sourced from the same counter that populates the `Usage` response body.
    /// Emitted once per completed request from the handler (both blocking and
    /// streaming paths), never from the engine decode loop.
    CompletionTokens(u32),
    /// F9: inter-token latency 99th percentile (ms), emitted alongside `ItlStats`.
    ///
    /// Separate event (rather than extending the `ItlStats` struct) so that
    /// the drainer can map it to a single `MetricEntry` row, matching the one
    /// row per event per metric convention used by `TtftMs`/`PromptTokens`.
    ItlP99Ms(f64),
    /// F9: count of inter-token latency spikes for this request.
    ///
    /// Spike = any interval > 3 × median (p50). Emitted alongside `ItlStats`
    /// and `ItlP99Ms` once per request after all decode steps complete.
    ItlSpikes(u64),
    /// SSD-tier: current on-disk footprint of the KV-block cache in bytes.
    ///
    /// Read from `SsdKvIndex::total_bytes()` at request boundary or on a tick.
    /// Emitted under `ssd_bytes_used`. Per-namespace; the `model_id` field on
    /// `MetricEvent` carries the namespace.
    SsdBytesUsed(u64),
    /// SSD-tier: delta count of LRU evictions since last emit.
    ///
    /// Incremented inside `evict_lru_until` and accumulated. Emitted under
    /// `ssd_evict_total`.
    SsdEvictTotal(u64),
    /// SSD-tier: per-spill timing aggregate for registry metrics.
    ///
    /// Carries the raw duration in ms for `ssd_spill_ms` and throughput for
    /// `ssd_spill_mb_per_s`. Real p50/p99 aggregation is performed by the
    /// Prometheus histogram in `openai.rs`; this event feeds raw SQLite rows.
    SsdSpillUs {
        /// Total spill duration (µs).
        dur_us: u64,
        /// Bytes written.
        bytes: u64,
    },
    /// SSD-tier: per-hydrate timing aggregate for registry metrics.
    SsdHydrateUs {
        /// Total hydrate duration (µs).
        dur_us: u64,
        /// Bytes read.
        bytes: u64,
    },
}

/// Bounded SPSC producer handle.
///
/// Cheaply cloneable — each clone shares the same channel and dropped counter.
/// `try_emit` is the only write path; it never blocks.
#[derive(Debug, Clone)]
pub struct DrainerHandle {
    /// `None` when observation recording is off — no task was spawned, so
    /// events are dropped at the producer and no `RunRecord` is ever built.
    tx: Option<mpsc::Sender<MetricEvent>>,
    /// Total events dropped due to channel full.
    dropped: Arc<AtomicU64>,
}

impl DrainerHandle {
    /// Attempt to send an event to the consumer.
    ///
    /// Returns `true` if enqueued. On channel full, increments the dropped
    /// counter and returns `false` — the caller is never blocked.
    /// Returns `false` immediately when metrics recording is disabled.
    pub fn try_emit(&self, event: MetricEvent) -> bool {
        let Some(tx) = self.tx.as_ref() else {
            return false;
        };
        match tx.try_send(event) {
            Ok(()) => true,
            Err(mpsc::error::TrySendError::Full(_)) => {
                let prev = self.dropped.fetch_add(1, Ordering::Relaxed);
                // Log once every 64 drops to avoid log spam.
                if prev.is_multiple_of(64) {
                    tracing::warn!(
                        dropped_total = prev + 1,
                        "metrics_drainer: channel full, event dropped"
                    );
                }
                false
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                // Drainer task has exited — treat as dropped.
                false
            }
        }
    }

    /// Number of events dropped since startup due to backpressure.
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

// ── Spawn ─────────────────────────────────────────────────────────────────────

/// Capacity of the bounded SPSC channel.
///
/// 256 events at ~200 bytes each ≈ 50 KB worst-case in-flight.
/// Normal server load (< 10 RPS) will never fill this.
const CHANNEL_CAPACITY: usize = 256;

/// Batch flush threshold (events).
const BATCH_SIZE: usize = 32;

/// Batch flush interval.
const FLUSH_INTERVAL_MS: u64 = 100;

/// Spawn the drainer background task and return the producer handle.
///
/// `db_path` — path to `metrics/runs.db`. Created (with migrations) if absent.
/// Run identity (backend, version, git sha, build profile, hardware tag) is
/// taken from `RunIdentity::rmlx()` — the drainer never invents it.
///
/// The task lives for the duration of the tokio runtime. When the last
/// `DrainerHandle` is dropped, the channel closes and the task drains remaining
/// events then exits.
///
/// Under any mode below `full` no task is spawned at all: the handle's sender is
/// `None`, `try_emit` drops at the producer, and the DB is never touched.
pub fn spawn_drainer(db_path: PathBuf) -> DrainerHandle {
    let dropped = Arc::new(AtomicU64::new(0));

    if !rmlx_metrics::mode::observations_enabled() {
        tracing::info!("metrics_drainer: observations disabled, task not spawned");
        return DrainerHandle { tx: None, dropped };
    }

    let (tx, rx) = mpsc::channel::<MetricEvent>(CHANNEL_CAPACITY);
    let dropped_task = Arc::clone(&dropped);

    tokio::spawn(drainer_task(rx, db_path, dropped_task));

    DrainerHandle {
        tx: Some(tx),
        dropped,
    }
}

// ── Consumer task ─────────────────────────────────────────────────────────────

async fn drainer_task(
    mut rx: mpsc::Receiver<MetricEvent>,
    db_path: PathBuf,
    dropped: Arc<AtomicU64>,
) {
    tracing::info!(
        db_path = %db_path.display(),
        "metrics_drainer: task started"
    );

    let mut batch: Vec<MetricEvent> = Vec::with_capacity(BATCH_SIZE);

    loop {
        // Collect up to BATCH_SIZE events, with a 100 ms timeout.
        let deadline =
            tokio::time::Instant::now() + std::time::Duration::from_millis(FLUSH_INTERVAL_MS);

        loop {
            if batch.len() >= BATCH_SIZE {
                break;
            }
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                // Got an event before timeout.
                Ok(Some(ev)) => batch.push(ev),
                // Channel closed — drain remaining and exit.
                Ok(None) => {
                    flush_batch(&batch, &db_path, &dropped).await;
                    tracing::info!("metrics_drainer: channel closed, task exiting");
                    return;
                }
                // Timeout with partial batch — flush what we have.
                Err(_timeout) => break,
            }
        }

        if !batch.is_empty() {
            flush_batch(&batch, &db_path, &dropped).await;
            batch.clear();
        }
    }
}

/// Write the batch to SQLite inside `spawn_blocking` so no executor thread blocks.
async fn flush_batch(batch: &[MetricEvent], db_path: &Path, dropped: &Arc<AtomicU64>) {
    if batch.is_empty() {
        return;
    }

    let owned_batch = batch.to_vec();
    let db_path = db_path.to_path_buf(); // owned PathBuf for the spawn_blocking closure

    let result =
        tokio::task::spawn_blocking(move || write_batch_to_db(&owned_batch, &db_path)).await;

    match result {
        Ok(Ok(n)) => {
            tracing::debug!(observations = n, "metrics_drainer: flushed batch to SQLite");
        }
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "metrics_drainer: SQLite write error");
        }
        Err(join_err) => {
            tracing::warn!(error = %join_err, "metrics_drainer: spawn_blocking panicked");
        }
    }

    let total_dropped = dropped.load(Ordering::Relaxed);
    if total_dropped > 0 {
        tracing::debug!(
            total_dropped,
            "metrics_drainer: cumulative events dropped due to backpressure"
        );
    }
}

/// Synchronous: open DB, write all events in the batch, return observation count.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
fn write_batch_to_db(
    batch: &[MetricEvent],
    db_path: &Path,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // Ensure parent dir exists.
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut conn = schema::open(db_path)?;
    migrate::run_pending(&mut conn)?;

    // `inserted_by` carries the binary's real version, not a bare literal —
    // the audit column is useless for triage without one.
    let inserted_by = RunIdentity::rmlx().inserted_by("rmlx-server");
    let mut recorder = Recorder::new(&mut conn, inserted_by);
    let mut total_obs = 0usize;

    for ev in batch {
        // SSD-tier events: update the Prometheus histogram accumulators in
        // openai.rs in addition to writing to SQLite.
        match &ev.kind {
            MetricKind::SsdSpillUs { dur_us, .. } => {
                crate::openai::record_ssd_spill_obs(*dur_us);
            }
            MetricKind::SsdHydrateUs { dur_us, .. } => {
                crate::openai::record_ssd_hydrate_obs(*dur_us);
            }
            MetricKind::SsdBytesUsed(bytes) => {
                crate::openai::update_ssd_bytes_used(&ev.model_id, *bytes);
            }
            MetricKind::SsdEvictTotal(delta) => {
                crate::openai::increment_ssd_evict_total(&ev.model_id, *delta);
            }
            _ => {}
        }

        let run = match event_to_run_record(ev) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(
                    model_id = %ev.model_id,
                    error = %e,
                    "metrics_drainer: could not build RunRecord for event, skipping"
                );
                continue;
            }
        };
        match recorder.record_run(&run) {
            Ok(outcome) => {
                total_obs += outcome.observation_ids.len();
            }
            Err(e) => {
                tracing::warn!(
                    model_id = %ev.model_id,
                    error = %e,
                    "metrics_drainer: record_run error for event, skipping"
                );
            }
        }
    }

    Ok(total_obs)
}

/// Build a `RunRecord` from a single `MetricEvent`.
///
/// One event → one observation row. Prompt is a fixed sentinel
/// `"(live_request)"` so the recorder can dedup via sha256 without knowing
/// the actual prompt body.
///
/// Identity, namespace/model split, weight-quant inference and kv-quant
/// canonicalization all come from the builder — this function supplies the
/// measurement and nothing else.
fn event_to_run_record(ev: &MetricEvent) -> rmlx_metrics::error::Result<RunRecord> {
    RunRecordBuilder::rmlx(
        &ev.model_id,
        &ev.kv_quant,
        ev.ctx_max, // real per-request effective_max_ctx threaded via MetricEvent
        PromptRef::ByBody {
            name: "live_request".into(),
            body: serde_json::Value::String("(live_request)".into()),
            notes: Some("per-request telemetry from SPSC drainer".into()),
            tokens_approx: None,
        },
    )?
    .ts_utc(ev.ts_utc.clone())
    .notes("spsc_drainer")
    .metrics(event_kind_to_metrics(&ev.kind))
    .build()
}

/// Map a `MetricKind` to the `MetricEntry` vec expected by `RunRecord`.
fn event_kind_to_metrics(kind: &MetricKind) -> Vec<MetricEntry> {
    match kind {
        MetricKind::KvCacheBytes(v) => vec![MetricEntry {
            name: "kv_cache_bytes".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::TtftMs(v) => vec![MetricEntry {
            name: "ttft_warm_ms".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::PromptCacheHits(v) => vec![MetricEntry {
            name: "prompt_cache_hits".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::PromptCacheMisses(v) => vec![MetricEntry {
            name: "prompt_cache_misses".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::PromptCacheBytes(v) => vec![MetricEntry {
            name: "prompt_cache_bytes".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::BlockHits(v) => vec![MetricEntry {
            name: "prompt_cache_block_hits".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::BlockMisses(v) => vec![MetricEntry {
            name: "prompt_cache_block_misses".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::PartialHits(v) => vec![MetricEntry {
            name: "prompt_cache_partial_hits".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::HotCacheHits(v) => vec![MetricEntry {
            name: "prompt_cache_hot_cache_hits".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::HotCacheEvictions(v) => vec![MetricEntry {
            name: "prompt_cache_hot_cache_evictions".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::SsdHits(v) => vec![MetricEntry {
            name: "prompt_cache_ssd_hits".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::LoadPhases {
            mmap_ms,
            dequant_ms,
            gpu_residency_ms,
            first_kernel_ready_ms,
            total_ms,
        } => vec![
            MetricEntry {
                name: "load_mmap_ms".into(),
                value: Some(*mmap_ms),
                stddev: None,
            },
            MetricEntry {
                name: "load_dequant_ms".into(),
                value: Some(*dequant_ms),
                stddev: None,
            },
            MetricEntry {
                name: "load_gpu_residency_ms".into(),
                value: Some(*gpu_residency_ms),
                stddev: None,
            },
            MetricEntry {
                name: "load_first_kernel_ready_ms".into(),
                value: Some(*first_kernel_ready_ms),
                stddev: None,
            },
            MetricEntry {
                name: "load_total_ms".into(),
                value: Some(*total_ms),
                stddev: None,
            },
        ],
        MetricKind::ItlStats {
            p50_ms,
            p95_ms,
            mean_ms,
            ..
        } => vec![
            MetricEntry {
                name: "itl_p50_ms".into(),
                value: Some(*p50_ms),
                stddev: None,
            },
            MetricEntry {
                name: "itl_p95_ms".into(),
                value: Some(*p95_ms),
                stddev: None,
            },
            MetricEntry {
                name: "step_ms_mean".into(),
                value: Some(*mean_ms),
                stddev: None,
            },
        ],
        MetricKind::QueueWaitMs(v) => vec![MetricEntry {
            name: "queue_wait_ms".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::QueueDepth(v) => vec![MetricEntry {
            name: "queue_depth".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::MetalPeakAllocMb(v) => vec![MetricEntry {
            name: "metal_peak_alloc_mb".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::PromptTokens(v) => vec![MetricEntry {
            name: "prompt_tokens_live".into(),
            value: Some(f64::from(*v)),
            stddev: None,
        }],
        MetricKind::CompletionTokens(v) => vec![MetricEntry {
            name: "completion_tokens_live".into(),
            value: Some(f64::from(*v)),
            stddev: None,
        }],
        MetricKind::ItlP99Ms(v) => vec![MetricEntry {
            name: "itl_p99_ms".into(),
            value: Some(*v),
            stddev: None,
        }],
        MetricKind::ItlSpikes(v) => vec![MetricEntry {
            name: "itl_spikes".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::SsdBytesUsed(v) => vec![MetricEntry {
            name: "ssd_bytes_used".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        MetricKind::SsdEvictTotal(v) => vec![MetricEntry {
            name: "ssd_evict_total".into(),
            value: Some(*v as f64),
            stddev: None,
        }],
        // Spill timing: emit one raw `ssd_spill_ms` observation per event.
        // Real percentiles (p50, p99) come from the Prometheus histogram
        // (`rmlx_ssd_spill_us_bucket{le=...}` in `openai.rs`); the registry row
        // here provides SQLite queryability of the raw duration. Emitting the
        // same value as both p50 and p99 (H2 fix) was misleading — a single-sample
        // distribution has no meaningful percentile breakdown.
        MetricKind::SsdSpillUs { dur_us, bytes } => {
            let dur_ms = *dur_us as f64 / 1000.0;
            let mb_per_s = if *dur_us > 0 {
                (*bytes as f64 / (1024.0 * 1024.0)) / (*dur_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            vec![
                MetricEntry {
                    name: "ssd_spill_ms".into(),
                    value: Some(dur_ms),
                    stddev: None,
                },
                MetricEntry {
                    name: "ssd_spill_mb_per_s".into(),
                    value: Some(mb_per_s),
                    stddev: None,
                },
            ]
        }
        // Hydrate timing: same rationale as spill above.
        MetricKind::SsdHydrateUs { dur_us, bytes } => {
            let dur_ms = *dur_us as f64 / 1000.0;
            let mb_per_s = if *dur_us > 0 {
                (*bytes as f64 / (1024.0 * 1024.0)) / (*dur_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            vec![
                MetricEntry {
                    name: "ssd_hydrate_ms".into(),
                    value: Some(dur_ms),
                    stddev: None,
                },
                MetricEntry {
                    name: "ssd_hydrate_mb_per_s".into(),
                    value: Some(mb_per_s),
                    stddev: None,
                },
            ]
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "metrics_drainer_tests.rs"]
mod tests;
