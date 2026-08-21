//! Shared server state — `AppState`, `LoadedModel`, SSD-tier histogram,
//! TTFT/ITL ring-buffer types, and API error counters.
//
// LOC-exempt: AppState + LoadedModel + ApiErrorCounters + SsdHistogram are
// cohesive lifecycle types; splitting fragments the single source of truth
// for slot semantics (keep-alive timer arming, decode-lease identity,
// cooperative evict, SSD-tier accounting). The 1000 LOC threshold was crossed
// when keep-alive, lease identity, and round-2 regression-defence fixes were
// added. A future cleanup can lift `SsdHistogram` + `SsdTierAccum` + their
// helpers (~160 LOC, independent of `AppState`) into a sibling module if the
// threshold pressure grows.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex as PLMutex, RwLock as PLRwLock};

use rmlx_metrics::events::EventRecorder;
use serde_json::Value;
use tokio::task::JoinHandle;

use crate::admission::AdmissionHandle;
use crate::engine::Generator;
use crate::keep_alive::{DecodeLease, DecodeLeaseGuard, KeepAlivePolicy};
use crate::metrics_drainer::DrainerHandle;
use crate::registry::ModelRegistry;
use crate::session_cache::SessionCache;

// ── TTS model cache type ──────────────────────────────────────────────────────

/// Cached TTS model pair: `(Arc<Mutex<TtsModel>>, Arc<TtsTokenizer>)`.
///
/// The inner `TtsModel` is wrapped in a `Mutex` because `synthesize()` takes
/// `&mut TtsModel` for lazy weight loading on the first synthesis call.
type TtsCache = PLRwLock<
    Option<(
        Arc<PLMutex<rmlx_audio::tts::TtsModel>>,
        Arc<rmlx_audio::tts::TtsTokenizer>,
    )>,
>;

// ── Type alias for the loader closure ────────────────────────────────────────

/// Factory that loads a model from disk given its snapshot path and logical id.
///
/// Injected at startup so tests can replace `ArchGenerator::from_snapshot`
/// with a lightweight stub (`NotReadyGenerator`).
pub type ModelLoader =
    Arc<dyn Fn(&std::path::Path, &str) -> rmlx_core::Result<Box<dyn Generator>> + Send + Sync>;

// ── Loaded model slot ─────────────────────────────────────────────────────────

/// A model loaded into the GPU slot.
///
/// Apple Silicon Metal context is exclusive per process — at most one model
/// occupies GPU memory at a time. This struct tracks the lifecycle.
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI/server internal state — field set is coupled to the single-GPU lifecycle contract; adding fields requires reviewing all construction sites"
)]
pub struct LoadedModel {
    /// Model identifier (path or registry id) used to load this slot.
    pub id: String,
    /// The loaded model instance implementing [`Generator`].
    pub model: Arc<dyn Generator>,
    /// Timestamp when the model was loaded into the GPU slot.
    pub loaded_at: Instant,
    /// Timestamp of the most recent request served by this model.
    pub last_used: Instant,
    /// A2: per-process effective max prompt-context length for this loaded
    /// model. Cached here at load time so the per-request guard in
    /// `/v1/chat/completions` and `/v1/messages` can read it without paying
    /// for a trait dispatch on every request. See `Generator::effective_max_ctx`.
    pub effective_max_ctx: usize,
    /// Active-decode lease for this slot.
    ///
    /// Counter incremented by `DecodeLeaseGuard::acquire` at the start of every
    /// generation (chat, embeddings, audio STT/TTS) and decremented on guard
    /// drop. The keep-alive timer checks `count() > 0` before tearing the
    /// model down and suppresses the unload while non-zero.
    pub decode_lease: DecodeLease,
    /// Cancel handle for the currently-armed unload timer.
    ///
    /// Held in a `Mutex` because the timer is replaced on every request
    /// (cancel-and-respawn). Wrapped in `Arc` so cloned `LoadedModel`s in
    /// hot paths still point at the same handle without extra plumbing.
    /// `None` when the policy is `Pin` (no timer armed).
    pub unload_handle: Arc<PLMutex<Option<JoinHandle<()>>>>,
    /// Effective keep-alive policy for this slot.
    ///
    /// Captured at load time from the precedence chain. Per-request reset
    /// may override this (request-field > env > flag > default) when an
    /// inbound request supplies a `keep_alive` body field on a native route.
    pub keep_alive: KeepAlivePolicy,
}

impl std::fmt::Debug for LoadedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadedModel")
            .field("id", &self.id)
            .field("loaded_at", &self.loaded_at)
            .finish()
    }
}

// ── SSD-tier Prometheus histogram state ──────────────────────────────────────
//
// Process-global accumulator for SSD spill and hydrate events. Updated via
// `record_ssd_spill_obs` / `record_ssd_hydrate_obs` (called from the drainer
// when it processes `MetricKind::SsdSpillUs` / `SsdHydrateUs` events and
// from the SSD-tier integration hook). Read by `render_prometheus`.
//
// Histogram buckets (µs): 100, 500, 1000, 5000, 10000, 50000, 100000,
// 500000, 1000000. Counts are cumulative per the Prometheus convention.
// `sum_us` and `count` are used for `_sum` / `_count` exposition lines.

/// Histogram bucket boundaries in microseconds (inclusive upper bound).
pub(crate) const HIST_BUCKETS_US: [u64; 9] = [
    100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000,
];

/// Per-metric histogram accumulator (spill or hydrate).
#[derive(Default, Clone)]
pub(crate) struct SsdHistogram {
    /// Cumulative count per bucket (index i = le = `HIST_BUCKETS_US[i]`).
    pub(crate) buckets: [u64; HIST_BUCKETS_US.len()],
    /// Count of observations that fell beyond the last finite bucket (overflow
    /// diagnostic only — NOT added to `count` or `+Inf` exposition to avoid
    /// double-counting; see H1 fix comment at the exposition sites).
    pub(crate) count_inf_overflow: u64,
    /// Sum of all observed durations in µs.
    pub(crate) sum_us: u64,
    /// Total observation count. Used directly as `_count` and `+Inf` bucket.
    pub(crate) count: u64,
}

impl SsdHistogram {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub(crate) fn observe(&mut self, dur_us: u64) {
        self.count += 1;
        self.sum_us += dur_us;
        let mut any = false;
        for (i, &le) in HIST_BUCKETS_US.iter().enumerate() {
            if dur_us <= le {
                // All buckets from i onward have le ≥ this observation's value,
                // so increment them all (Prometheus cumulative convention:
                // bucket[i] = count of observations where obs ≤ le[i]).
                for b in &mut self.buckets[i..] {
                    *b += 1;
                }
                any = true;
                break;
            }
        }
        if !any {
            // Beyond the last finite bucket — the observation is counted by
            // `self.count` (the +Inf and _count exposition paths). Track the
            // overflow separately for diagnostics without double-counting.
            self.count_inf_overflow += 1;
        }
    }
}

/// Process-global SSD-tier accumulator.
#[derive(Default)]
pub(crate) struct SsdTierAccum {
    pub(crate) spill: SsdHistogram,
    pub(crate) hydrate: SsdHistogram,
    /// namespace → last-known on-disk footprint in bytes (O(1) lookup).
    pub(crate) bytes_used: HashMap<String, u64>,
    /// namespace → lifetime eviction counter (O(1) lookup).
    pub(crate) evict_total: HashMap<String, u64>,
}

use std::sync::LazyLock;

pub(crate) static SSD_TIER_ACCUM: LazyLock<PLMutex<SsdTierAccum>> =
    LazyLock::new(|| PLMutex::new(SsdTierAccum::default()));

/// Record one spill observation into the process-global SSD histogram.
///
/// Called by the drainer when processing a `MetricKind::SsdSpillUs` event.
/// Bytes are tracked by the SQLite observation row, not by this histogram.
pub(crate) fn record_ssd_spill_obs(dur_us: u64) {
    // parking_lot::Mutex never poisons; .lock() always succeeds.
    SSD_TIER_ACCUM.lock().spill.observe(dur_us);
}

/// Record one hydrate observation into the process-global SSD histogram.
///
/// Called by the drainer when processing a `MetricKind::SsdHydrateUs` event.
/// Bytes are tracked by the SQLite observation row, not by this histogram.
pub(crate) fn record_ssd_hydrate_obs(dur_us: u64) {
    SSD_TIER_ACCUM.lock().hydrate.observe(dur_us);
}

/// Register Prometheus observation hooks on the models layer.
///
/// Installs four closures:
/// - `set_ssd_spill_prom_hook` → calls `record_ssd_spill_obs` (histogram)
/// - `set_ssd_hydrate_prom_hook` → calls `record_ssd_hydrate_obs` (histogram)
/// - `set_ssd_bytes_used_hook` → calls `update_ssd_bytes_used` (gauge)
/// - `set_ssd_evict_total_hook` → calls `increment_ssd_evict_total` (counter)
///
/// Must be called at server startup, before any model load (spill threads spawn
/// during model load). Subsequent calls are no-ops (OnceLock first-writer wins).
pub fn register_ssd_prom_hooks() {
    // These hooks are called only from the synchronous drainer, never from an async handler.
    use std::sync::Arc;
    rmlx_kv_ssd::set_ssd_spill_prom_hook(Arc::new(|dur_us, _bytes| {
        record_ssd_spill_obs(dur_us);
    }));
    rmlx_kv_ssd::set_ssd_hydrate_prom_hook(Arc::new(|dur_us, _bytes| {
        record_ssd_hydrate_obs(dur_us);
    }));
    rmlx_kv_ssd::set_ssd_bytes_used_hook(Arc::new(|namespace, bytes| {
        update_ssd_bytes_used(namespace, bytes);
    }));
    rmlx_kv_ssd::set_ssd_evict_total_hook(Arc::new(|namespace, count| {
        increment_ssd_evict_total(namespace, count);
    }));
}

/// Update the on-disk byte gauge for a namespace.
///
/// Called from the SSD-tier wiring after `SsdKvIndex::total_bytes()`.
pub(crate) fn update_ssd_bytes_used(namespace: &str, bytes: u64) {
    SSD_TIER_ACCUM
        .lock()
        .bytes_used
        .insert(namespace.to_owned(), bytes);
}

/// Increment the lifetime eviction counter for a namespace.
///
/// Called from `evict_lru_until` or its wiring layer.
pub(crate) fn increment_ssd_evict_total(namespace: &str, delta: u64) {
    *SSD_TIER_ACCUM
        .lock()
        .evict_total
        .entry(namespace.to_owned())
        .or_insert(0) += delta;
}

/// Snapshot of the SSD-tier accumulator for Prometheus exposition.
pub(crate) struct SsdTierSnapshot {
    pub(crate) spill: SsdHistogram,
    pub(crate) hydrate: SsdHistogram,
    pub(crate) bytes_used: HashMap<String, u64>,
    pub(crate) evict_total: HashMap<String, u64>,
}

pub(crate) fn snapshot_ssd_tier() -> SsdTierSnapshot {
    let g = SSD_TIER_ACCUM.lock();
    SsdTierSnapshot {
        spill: g.spill.clone(),
        hydrate: g.hydrate.clone(),
        bytes_used: g.bytes_used.clone(),
        evict_total: g.evict_total.clone(),
    }
}

/// One TTFT sample stored in the rolling ring buffer.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI wire/metrics DTO — field set tracks upstream metric spec; exhaustiveness is the contract"
)]
pub struct TtftSample {
    /// Identifier of the model that produced this sample.
    pub model_id: String,
    /// Time-to-first-token latency in milliseconds.
    pub ttft_ms: u64,
}

/// Rolling ring-buffer of the last N TTFT samples across all models.
///
/// Capped at `TTFT_RING_CAPACITY` entries. Oldest entry is evicted when full.
pub type TtftStore = Arc<PLMutex<VecDeque<TtftSample>>>;

/// Maximum number of TTFT samples kept in memory.
pub const TTFT_RING_CAPACITY: usize = 20;

/// Per-request ITL (inter-token latency) aggregate sample (M30).
///
/// Written by the blocking decode thread after all steps complete.
/// Read by `GET /metrics/cache` to populate the `last_itl` block.
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI wire/metrics DTO — field set tracks upstream metric spec; exhaustiveness is the contract"
)]
pub struct ItlSample {
    /// Logical model id this sample belongs to.
    pub model_id: String,
    /// Median inter-token latency in milliseconds.
    pub p50_ms: f64,
    /// 95th-percentile inter-token latency in milliseconds.
    pub p95_ms: f64,
    /// Mean inter-token latency in milliseconds.
    pub mean_ms: f64,
    /// Number of decode steps measured.
    pub step_count: usize,
}

/// Rolling ring-buffer of the last N ITL samples across all models.
///
/// Same pattern as `TtftStore`. Capped at `ITL_RING_CAPACITY` entries.
pub type ItlStore = Arc<PLMutex<VecDeque<ItlSample>>>;

/// Maximum number of ITL samples kept in memory.
pub const ITL_RING_CAPACITY: usize = 20;

// ── F8: API error-category lifetime counters ─────────────────────────────────

/// HTTP-boundary API error categories.
///
/// Classified at the site that already maps an error or condition to an HTTP
/// response — no re-derivation from `rmlx_core::Error` variants beyond what
/// the existing mappers already match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "OpenAI API wire category — adding a variant requires matching dispatch in ApiErrorCounters and all increment sites"
)]
pub enum ApiErrorCategory {
    /// HTTP 400 `invalid_request_error` — bad request field, out-of-range
    /// param, unsupported field, etc.
    BadRequest,
    /// HTTP 400 `context_length_exceeded` — A2 prompt-length guard.
    ContextOverflow,
    /// HTTP 404 `not_found_error` / `model_not_found` — model absent from
    /// registry.
    NotFound,
    /// HTTP 507 `oom_during_load` — weight-load OOM (J3).
    OomLoad,
    /// HTTP 507 `oom_kv_cache` — KV-cache allocation OOM (J3).
    OomKvCache,
    /// HTTP 503 `oom_mid_stream` — mid-decode OOM (J3).
    OomMidStream,
    /// HTTP 408 `timeout` — A8 per-request wall-clock timeout.
    Timeout,
    /// HTTP 503 `service_unavailable` — loader failure, missing pipeline,
    /// engine catch-all (non-OOM).
    Upstream,
    /// HTTP 500 `internal_error` — NaN logits / smoke probe / task panic.
    Internal,
    /// HTTP 429 `rate_limit_error` — C5 admission-queue full.
    RateLimit,
    /// HTTP 503 `admission_sla_exceeded` — anticipatory SLA rejection.
    ///
    /// Distinct from `Upstream` (catch-all engine errors) so dashboards can
    /// track admission-controller rejections independently from actual engine
    /// failures. Counter label: `"admission_sla_503"`.
    AdmissionSla503,
}

impl ApiErrorCategory {
    /// Stable snake_case label used in the `/metrics/cache` JSON key.
    pub fn as_str(self) -> &'static str {
        match self {
            ApiErrorCategory::BadRequest => "bad_request",
            ApiErrorCategory::ContextOverflow => "context_overflow",
            ApiErrorCategory::NotFound => "not_found",
            ApiErrorCategory::OomLoad => "oom_load",
            ApiErrorCategory::OomKvCache => "oom_kv_cache",
            ApiErrorCategory::OomMidStream => "oom_mid_stream",
            ApiErrorCategory::Timeout => "timeout",
            ApiErrorCategory::Upstream => "upstream",
            ApiErrorCategory::Internal => "internal",
            ApiErrorCategory::RateLimit => "rate_limit",
            ApiErrorCategory::AdmissionSla503 => "admission_sla_503",
        }
    }
}

/// Process-lifetime per-category error counters (F8).
///
/// One `Arc<AtomicU64>` per `ApiErrorCategory` variant. Stored on `AppState`
/// and incremented at every HTTP error-response emission site. Exposed as
/// `error_counts` in `GET /metrics/cache`.
///
/// Storage layout mirrors F14 (`tokens_in` / `tokens_out`): plain
/// `Arc<AtomicU64>` fields, constructed with `AtomicU64::new(0)`.
#[derive(Clone, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI server internal — field set mirrors ApiErrorCategory variants; adding a field requires a matching variant and all increment sites"
)]
pub struct ApiErrorCounters {
    /// Counter for `ApiErrorCategory::BadRequest` errors.
    pub bad_request: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::ContextOverflow` errors.
    pub context_overflow: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::NotFound` errors.
    pub not_found: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::OomLoad` errors.
    pub oom_load: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::OomKvCache` errors.
    pub oom_kv_cache: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::OomMidStream` errors.
    pub oom_mid_stream: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::Timeout` errors.
    pub timeout: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::Upstream` errors.
    pub upstream: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::Internal` errors.
    pub internal: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::RateLimit` errors.
    pub rate_limit: Arc<std::sync::atomic::AtomicU64>,
    /// Counter for `ApiErrorCategory::AdmissionSla503` errors.
    pub admission_sla_503: Arc<std::sync::atomic::AtomicU64>,
}

impl ApiErrorCounters {
    /// Construct all counters starting at zero.
    pub fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        Self {
            bad_request: Arc::new(AtomicU64::new(0)),
            context_overflow: Arc::new(AtomicU64::new(0)),
            not_found: Arc::new(AtomicU64::new(0)),
            oom_load: Arc::new(AtomicU64::new(0)),
            oom_kv_cache: Arc::new(AtomicU64::new(0)),
            oom_mid_stream: Arc::new(AtomicU64::new(0)),
            timeout: Arc::new(AtomicU64::new(0)),
            upstream: Arc::new(AtomicU64::new(0)),
            internal: Arc::new(AtomicU64::new(0)),
            rate_limit: Arc::new(AtomicU64::new(0)),
            admission_sla_503: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment the counter for `category` by 1 (Relaxed ordering).
    pub fn increment(&self, category: ApiErrorCategory) {
        use std::sync::atomic::Ordering::Relaxed;
        match category {
            ApiErrorCategory::BadRequest => self.bad_request.fetch_add(1, Relaxed),
            ApiErrorCategory::ContextOverflow => self.context_overflow.fetch_add(1, Relaxed),
            ApiErrorCategory::NotFound => self.not_found.fetch_add(1, Relaxed),
            ApiErrorCategory::OomLoad => self.oom_load.fetch_add(1, Relaxed),
            ApiErrorCategory::OomKvCache => self.oom_kv_cache.fetch_add(1, Relaxed),
            ApiErrorCategory::OomMidStream => self.oom_mid_stream.fetch_add(1, Relaxed),
            ApiErrorCategory::Timeout => self.timeout.fetch_add(1, Relaxed),
            ApiErrorCategory::Upstream => self.upstream.fetch_add(1, Relaxed),
            ApiErrorCategory::Internal => self.internal.fetch_add(1, Relaxed),
            ApiErrorCategory::RateLimit => self.rate_limit.fetch_add(1, Relaxed),
            ApiErrorCategory::AdmissionSla503 => self.admission_sla_503.fetch_add(1, Relaxed),
        };
    }

    /// Snapshot all counters as a `serde_json::Value` object for `/metrics/cache`.
    pub fn to_json(&self) -> Value {
        use std::sync::atomic::Ordering::Relaxed;
        serde_json::json!({
            "bad_request":        self.bad_request.load(Relaxed),
            "context_overflow":   self.context_overflow.load(Relaxed),
            "not_found":          self.not_found.load(Relaxed),
            "oom_load":           self.oom_load.load(Relaxed),
            "oom_kv_cache":       self.oom_kv_cache.load(Relaxed),
            "oom_mid_stream":     self.oom_mid_stream.load(Relaxed),
            "timeout":            self.timeout.load(Relaxed),
            "upstream":           self.upstream.load(Relaxed),
            "internal":           self.internal.load(Relaxed),
            "rate_limit":         self.rate_limit.load(Relaxed),
            "admission_sla_503":  self.admission_sla_503.load(Relaxed),
        })
    }
}

impl Default for ApiErrorCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared server state.
///
/// `slots` holds up to `max_loaded_models` `LoadedModel`s concurrently.
/// Apple Silicon Metal context is exclusive per process, so multiple models
/// may be resident in memory but only one forward pass runs at a time — that
/// is enforced by `gpu_gate`, a single process-wide mutex injected into every
/// generator's serialisation lock.
///
/// `loader` is the function used to create a new `Generator` from a snapshot
/// path — injected at startup so tests can swap it for `NotReadyGenerator`.
#[derive(Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI server AppState — internal type constructed only in run_serve; adding a field requires updating the single construction site"
)]
pub struct AppState {
    /// Registry of all known model snapshots and their metadata.
    pub registry: Arc<ModelRegistry>,
    /// Resident model slots. Length ≤ `max_loaded_models`. An empty Vec =
    /// no model loaded. At `max_loaded_models == 1` this behaves exactly
    /// like the old single-slot `Option<LoadedModel>` (swap-on-different-id).
    pub slots: Arc<PLRwLock<Vec<LoadedModel>>>,
    /// jina-embeddings-v4 lazily-loaded embedding model.
    ///
    /// Encoder, NOT a `Generator` — kept out of `slots` (every `slots` entry
    /// is a causal LM). At most one embedding model is resident (Metal is
    /// single-process). Loaded on first `/v1/embeddings` request whose model
    /// id resolves to a `JinaEmbeddingsV4Model` registry entry, and reused
    /// (with a per-request `apply_task` swap under the GPU gate) thereafter.
    pub embed_slot: Arc<PLRwLock<Option<crate::embeddings::JinaEmbedModel>>>,
    /// Shared multimodal encoder-output cache. One instance per
    /// `AppState`, threaded into every vision/audio generator and embedding
    /// path. Sized by `--mm-cache-bytes` (default 512 MiB; `0` disables).
    pub mm_cache: Arc<rmlx_models::multimodal_cache::MultimodalCache>,
    /// Process-wide GPU serialisation gate. A clone of this `Arc` is threaded
    /// into every loaded generator's `_lock` field so the existing
    /// try_lock/warn/lock critical section in `Generator::generate` serialises
    /// across ALL resident models (single Metal context per process).
    pub gpu_gate: Arc<PLMutex<()>>,
    /// C5 Slice A: process-wide FIFO admission gate over the single-GPU
    /// serialisation. Constructed with exactly **1 permit** — `tokio`'s
    /// `Semaphore` hands out permits in strict FIFO arrival order, which is
    /// the fairness fix (the old `gpu_gate` `try_lock`→`lock` path acquired
    /// in OS-arbitrary order under contention). A request `acquire_owned`s
    /// the single permit before generation and holds it (RAII) for the whole
    /// decode; the next waiter is admitted only when it drops. Throughput is
    /// unchanged — still one forward at a time; this adds fairness +
    /// bounded-depth rejection + queue observability only.
    pub gpu_queue: Arc<tokio::sync::Semaphore>,
    /// C5 Slice A: count of requests that have passed the depth check and
    /// not yet finished (admitted-and-in-flight, including the one holding
    /// the permit plus those waiting on `acquire_owned`). Used both as the
    /// bound for the `max_queue_depth` 429 reject and as the `queue_depth`
    /// gauge emitted to metrics at permit-acquire. Decremented by an RAII
    /// guard on every completion path (success, error, timeout, drop).
    pub gpu_pending: Arc<AtomicUsize>,
    /// C5 Slice A: maximum number of admitted-and-in-flight requests before
    /// new ones are rejected with HTTP 429 (`server queue full`). `0` =
    /// unlimited (no admission rejection, FIFO + metrics still apply).
    /// Configurable via `--max-queue-depth` (default 64).
    pub max_queue_depth: usize,
    /// Maximum number of models resident at once. Default 1 (byte-equivalent
    /// to the old single-slot behaviour). When the slots are full and a
    /// different model is requested, the least-recently-used entry is evicted.
    pub max_loaded_models: usize,
    /// Factory for loading a model by snapshot path.
    ///
    /// Signature: `(path, id) -> Result<Box<dyn Generator>>`.
    /// Production uses `ArchGenerator::from_snapshot`; tests inject a stub.
    pub loader: ModelLoader,
    /// Optional metrics sink. `None` in unit tests; `Some` in production runs.
    pub metrics: Option<Arc<EventRecorder>>,
    /// Server-startup keep-alive policy applied to every freshly
    /// loaded model (overridable by the per-request `keep_alive` field on
    /// native routes). When this is `KeepAlivePolicy::Pin` no timer is armed
    /// and the model stays resident until the process exits or it is
    /// explicitly unloaded.
    ///
    /// Default (when no CLI flag, env var, or request field is set): 15 min.
    /// CLI: `--idle-timeout-secs` (accepts `0` / `-1` / `30s` / `15m` / `2h`).
    pub idle_policy: KeepAlivePolicy,
    /// Per-request `max_tokens` ceiling. Requests exceeding this return HTTP 400
    /// (`invalid_request_error`). Default `u32::MAX` (no cap). Configurable via
    /// the `--max-tokens-cap` CLI flag. Replaces the Stage-1 hardcoded cap of 64.
    pub max_tokens_cap: u32,
    /// Server-startup cap on per-request wall-clock timeout, in seconds.
    ///
    /// Default `600`. Configurable via `--max-timeout-secs`. Applied by the
    /// `timeout_mw` axum middleware on every request: the middleware wraps the
    /// whole handler future (including SSE stream) in `tokio::time::timeout`.
    /// Per-request `X-Request-Timeout-Seconds` header can lower the effective
    /// timeout, but never exceed this cap. 0 = no timeout (disabled).
    pub max_timeout_secs: u64,
    /// Per-session KV-reuse registry (N2).
    ///
    /// Tracks active `X-Session-Id` → prompt_len entries. The route handler
    /// calls `touch()` on each request to update last_used and derive the
    /// effective `prompt_cache_slots` (base + active_count) passed to
    /// `generate_greedy` via `GenerationRequest::effective_prompt_cache_slots`.
    pub session_cache: Arc<PLMutex<SessionCache>>,
    /// Prompt-cache slots this server was configured with
    /// (`--prompt-cache-slots`), before any per-request session adjustment.
    ///
    /// The session-KV-reuse path needs it for two reasons. It has to widen the
    /// operator's setting rather than replace it — a hard-coded base gives
    /// someone who asked for 8 slots a 4-slot cache. And `0` means the cache is
    /// disabled, which a request header must not be able to undo: re-enabling
    /// it would store snapshots on a server configured to store none, and the
    /// alternation between the two capacities would rebuild the cache on every
    /// request.
    pub prompt_cache_slots: usize,
    /// Rolling ring-buffer of per-request TTFT samples (L6).
    ///
    /// Written by `generate_streaming` when the first token arrives from the
    /// decode thread; read by `GET /metrics/cache`. Capped at
    /// `TTFT_RING_CAPACITY` entries — oldest evicted when full.
    pub ttft_store: TtftStore,
    /// Rolling ring-buffer of per-request ITL aggregate samples (M30).
    ///
    /// Written by the blocking decode thread after all tokens are produced;
    /// read by `GET /metrics/cache` to populate the `last_itl` block.
    /// Capped at `ITL_RING_CAPACITY` entries — oldest evicted when full.
    pub itl_store: ItlStore,
    /// SPSC async drainer for per-request SQLite metrics (F6/L18).
    ///
    /// `None` when no drainer has been started (e.g. unit-test stubs that
    /// do not need SQLite persistence). Production `run_serve` always sets this.
    pub metrics_drainer: Option<DrainerHandle>,
    /// B5: when `true`, run the 8-token smoke probe on first model load and
    /// refuse to serve if the verdict is `BrokenPunctLoop` or `BrokenNan`.
    /// Default `false` (zero-overhead path unchanged).
    pub require_smoke_probe: bool,
    /// G4: server-startup default temperature applied when the request omits
    /// `temperature`. Precedence: request > this > model generation_defaults >
    /// hard-coded 1.0. `None` = absent (behaviour unchanged).
    /// Configurable via `--default-temperature`.
    pub default_temperature: Option<f32>,
    /// server-startup default for `enable_thinking` (thinking mode control).
    ///
    /// `Some(false)` → suppress the open `<think>` block on Qwen3-family models
    /// unless the request explicitly overrides with `enable_thinking: true`.
    /// `None` = absent (behaviour unchanged — thinking enabled by template default).
    /// Configurable via `--enable-thinking false` / `--enable-thinking true`.
    ///
    /// Precedence: request `enable_thinking` > this > absent (= enabled).
    pub default_enable_thinking: Option<bool>,
    /// Server-startup default image-token budget for Gemma4-unified vision.
    /// `Some(n)` raises the soft-token budget for dense images (clamped to the
    /// model's safe upper bound by the preprocessor). `None` = absent (use the
    /// snapshot's `processor_config.json` `max_soft_tokens`). Configurable via
    /// `--image-max-tokens`.
    ///
    /// Precedence: request `image_max_tokens` > this > snapshot config default.
    pub default_image_max_tokens: Option<usize>,
    /// F14: process-lifetime cumulative prompt (input) token counter.
    ///
    /// Incremented at the same request-completion site that emits
    /// `MetricKind::PromptTokens` to the SPSC drainer (single source, no
    /// double-count). Exposed as `tokens_in` in `GET /metrics/cache`.
    pub tokens_in: Arc<std::sync::atomic::AtomicU64>,
    /// F14: process-lifetime cumulative completion (output) token counter.
    ///
    /// Incremented at the same request-completion site that emits
    /// `MetricKind::CompletionTokens` to the SPSC drainer. Exposed as
    /// `tokens_out` in `GET /metrics/cache`.
    pub tokens_out: Arc<std::sync::atomic::AtomicU64>,
    /// F8: per-category HTTP error lifetime counters.
    ///
    /// Incremented at every HTTP error-response emission site (OpenAI +
    /// Anthropic routes, timeout middleware). Exposed as `error_counts` in
    /// `GET /metrics/cache`.
    pub error_counts: ApiErrorCounters,
    /// wall-clock instant at which this AppState was constructed (= server startup).
    ///
    /// Used to compute `uptime_s` in `GET /v1/metrics`. Stored as `Instant`
    /// (monotonic, no calendar drift) — never serialised directly.
    pub started_at: Instant,
    /// process-lifetime count of requests that were admitted (passed
    /// the depth check and entered the FIFO queue). Incremented once per
    /// `Admission::Admitted` return from `admit_request`. Exposed as
    /// `requests_started` in `GET /v1/metrics`.
    pub requests_started: Arc<std::sync::atomic::AtomicU64>,
    /// process-lifetime count of requests that completed successfully
    /// (non-streaming: response serialised; streaming: stream exhausted with
    /// no error). Incremented at the end of `generate_blocking` /
    /// `generate_streaming` on the success path. Exposed as
    /// `requests_completed` in `GET /v1/metrics`.
    pub requests_completed: Arc<std::sync::atomic::AtomicU64>,
    /// process-lifetime count of requests that failed with an engine
    /// error (non-streaming: error response returned; streaming: first token
    /// was an error). Incremented at the engine-error return sites.
    /// Exposed as `requests_failed` in `GET /v1/metrics`.
    pub requests_failed: Arc<std::sync::atomic::AtomicU64>,

    /// Optional adaptive admission controller handle.
    ///
    /// `None` when `--adaptive-admission` is absent (default). When `None`,
    /// the existing open-loop FIFO semaphore in `engine::admit_request` is the
    /// sole admission path and behaviour is byte-identical to without the controller.
    ///
    /// When `Some`, the controller is enabled. The active `max_queue_depth` used
    /// by `admit_request` is read from `controller.current_depth` (atomic) on
    /// every request, overriding the static `AppState::max_queue_depth`.
    /// `max_queue_depth` is preserved as the initial value and hard floor.
    pub admission_controller: Option<crate::admission::ControllerHandle>,
    /// RAII handle for the admission controller background tick task.
    ///
    /// `None` when `--adaptive-admission` is absent. When `Some`, the last
    /// `Arc` clone drop (on runtime teardown) aborts the tick loop within at
    /// most 1 s via `AdmissionHandle::drop → JoinHandle::abort`.
    ///
    /// Wrapped in `Arc` so `AppState::clone` is cheap (clones share the same
    /// task handle; the task is aborted when all clones are gone).
    pub admission_handle: Option<Arc<AdmissionHandle>>,

    /// Optional path to the Whisper model snapshot directory.
    ///
    /// Set via `--whisper-model-path` or `RMLX_WHISPER_MODEL_PATH`.
    /// `None` → audio endpoints return 503.
    pub whisper_model_path: Option<std::path::PathBuf>,
    /// Optional path to the Whisper tokenizer directory.
    ///
    /// Set via `--whisper-tokenizer-path` or `RMLX_WHISPER_TOKENIZER_PATH`.
    /// The mlx-community Whisper snapshot does NOT ship tokenizer files;
    /// place the openai/whisper-large-v3 tokenizer alongside the snapshot,
    /// or point here. `None` → audio endpoints return 503.
    pub whisper_tokenizer_path: Option<std::path::PathBuf>,
    /// Lazily-loaded Whisper model + tokenizer cache.
    ///
    /// Populated on the first successful audio request. Subsequent requests
    /// `read()` the `Option` and `Arc::clone` — no re-load, no re-parse.
    /// Load is serialised through the GPU admission gate (it's GPU work):
    /// the first request loads while holding the semaphore permit; later
    /// requests see the populated cache under the same permit flow.
    ///
    /// Keyed implicitly by `whisper_model_path` (one server, one snapshot).
    /// Changing the snapshot path requires a server restart.
    pub audio_model: Arc<
        PLRwLock<
            Option<(
                Arc<rmlx_audio::whisper::WhisperModel>,
                Arc<rmlx_audio::tokenizer::WhisperTokenizer>,
            )>,
        >,
    >,
    /// Optional path to the Qwen3-TTS model snapshot.
    ///
    /// Set via `--tts-model-path` or `RMLX_TTS_MODEL_PATH`.
    /// `None` → `/v1/audio/speech` returns 503.
    pub tts_model_path: Option<std::path::PathBuf>,
    /// Optional path to the Qwen3-TTS tokenizer (codec decoder) snapshot.
    ///
    /// Set via `--tts-tokenizer-path` or `RMLX_TTS_TOKENIZER_PATH`.
    pub tts_tokenizer_path: Option<std::path::PathBuf>,
    /// Lazily-loaded Qwen3-TTS model + tokenizer cache.
    ///
    /// Populated on the first successful TTS request. See [`TtsCache`].
    pub tts_model: Arc<TtsCache>,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("registry_len", &self.registry.list().len())
            .field("ttft_samples", &self.ttft_store.lock().len())
            .field("itl_samples", &self.itl_store.lock().len())
            .field(
                "metrics_drainer_dropped",
                &self
                    .metrics_drainer
                    .as_ref()
                    .map_or(0, DrainerHandle::dropped_count),
            )
            .finish()
    }
}

impl AppState {
    /// Effective max prompt-context length for the resident model `model_id`.
    ///
    /// Returns `usize::MAX` when the model is not resident (cold-start race)
    /// or when the generator does not participate in KV-cache sizing — the
    /// existing 503 path then catches real runtime overflows. Used by the
    /// A2 `context_length_exceeded` guard in the chat routes.
    pub fn effective_max_ctx_for(&self, model_id: &str) -> usize {
        self.slots
            .read()
            .iter()
            .find(|m| m.id == model_id)
            .map_or(usize::MAX, |m| m.effective_max_ctx)
    }

    /// Load model `id` into a resident slot.
    ///
    /// If the model is already resident, updates its `last_used` and returns
    /// without reloading. Otherwise, if there is free slot capacity it is
    /// loaded and pushed; if the slots are full the least-recently-used entry
    /// is evicted (mirroring the old single-slot swap at `max_loaded_models
    /// == 1`) before the new model is loaded.
    ///
    /// Returns a clone of the `Arc<dyn Generator>` for the caller to use,
    /// plus a `bool` that is `true` when the model was just loaded (cold) and
    /// `false` when it was already resident (warm).
    pub fn ensure_loaded(&self, model_id: &str) -> Result<(Arc<dyn Generator>, bool), String> {
        let entry = self
            .registry
            .get(model_id)
            .ok_or_else(|| format!("model '{model_id}' not found in registry"))?;

        // First, check resident / collect any LRU eviction target under the
        // slots write lock. We finalize the eviction OUTSIDE the lock so
        // (a) the slots lock isn't held across the finalize_unload session-
        // cache lock acquisition, and (b) the evicted slot's
        // `Arc<dyn Generator>` is dropped BEFORE the new model loader runs,
        // releasing GPU residency first.
        let evicted_for_cleanup: Option<LoadedModel> = {
            let mut slots = self.slots.write();

            // Already resident?
            if let Some(loaded) = slots.iter_mut().find(|m| m.id == model_id) {
                loaded.last_used = Instant::now();
                // Swap the slot's decode_lease so the previous
                // timer's identity token (Arc::as_ptr of the old lease) no
                // longer matches the slot's lease pointer. Any stale TTL
                // fire from the prior arm that already passed its sleep
                // boundary will detect the mismatch under the write lock
                // and bail (see `arm_or_reset_timer` for the check).
                //
                // Safe to swap here: at this point no decode_lease_guard
                // for THIS request has been acquired yet — the handler
                // calls `decode_lease_guard` only after `ensure_loaded`
                // returns. Pending guards from prior in-flight requests
                // retain their own `Arc` clones (the swap only changes the
                // slot's "live" lease pointer; old guards still decrement
                // the captured `Arc` they hold). The new timer's busy
                // check uses the new lease — it cannot observe in-flight
                // decodes that still hold the old `Arc`, but those decodes
                // are kept alive by `Arc<dyn Generator>` already in the
                // handler's hand: even if a stale unload runs, the
                // in-flight response still completes correctly.
                loaded.decode_lease = Arc::new(AtomicUsize::new(0));
                let gen = Arc::clone(&loaded.model);
                let policy = loaded.keep_alive;
                let lease = Arc::clone(&loaded.decode_lease);
                let handle_slot = Arc::clone(&loaded.unload_handle);
                drop(slots);
                // Reset the timer on every request (cancel + respawn).
                self.arm_or_reset_timer(
                    model_id,
                    policy,
                    &lease,
                    &handle_slot,
                    /*reset=*/ true,
                );
                return Ok((gen, false));
            }

            // Not resident. If at capacity, evict the LRU entry first (this is
            // the swap path; at max_loaded_models == 1 it evicts the single
            // existing entry, byte-equivalent to the old `*slot = None`).
            //
            // This is the cooperative same-process evict path — when a
            // different model is requested while another is resident, the
            // current resident is unloaded regardless of its remaining TTL
            // (LM Studio "Auto-Evict" semantics). The TTL on the new model is
            // armed below. Finalisation (handle abort + session-cache + tracing)
            // happens through `finalize_unload(.., CooperativeEvict)` after the
            // write lock is released, so there is a single tracing site per
            // unload reason.
            if slots.len() >= self.max_loaded_models {
                // Index of the entry with the minimum `last_used`.
                slots
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, m)| m.last_used)
                    .map(|(idx, _)| idx)
                    .map(|idx| slots.swap_remove(idx))
            } else {
                None
            }
        };
        if let Some(evicted) = evicted_for_cleanup {
            self.finalize_unload(&evicted, UnloadReason::CooperativeEvict, Some(model_id));
            // Explicit drop: the Arc<dyn Generator> is released here so its
            // GPU residency is freed before the new model loader runs below.
            drop(evicted);
        }

        // B5: --require-smoke-probe gate.
        // Run before the generator is placed in the slot so a broken snapshot
        // is never served. Default-OFF (require_smoke_probe=false) keeps the
        // zero-overhead path unchanged.
        if self.require_smoke_probe {
            tracing::info!(model_id, path = %entry.abs_path.display(), "B5: running smoke probe before first load");

            // Render the smoke seed through the model's real chat template so the
            // probe exercises production-shaped, turn-structured input. When no
            // usable template exists, `smoke_prompt_ids` returns None and
            // `run_smoke_probe` builds the bare seed itself with its own
            // canonical BOS resolution — no token id is invented here.
            let templated_prompt = crate::tokenizer_io::load_tokenizer(&entry.abs_path)
                .ok()
                .and_then(|tk| crate::chat_template::smoke_prompt_ids(&entry.abs_path, &tk));

            let verdict = rmlx_models::arch::run_smoke_probe(
                &entry.abs_path,
                rmlx_mlx::Device::Gpu,
                None, // use the engine default KV quant
                None, // use model default max_ctx
                templated_prompt,
            )
            .map_err(|e| format!("smoke probe error for '{model_id}': {e}"))?;

            use rmlx_models::SmokeVerdict;
            match &verdict {
                SmokeVerdict::Ok | SmokeVerdict::Inconclusive { .. } => {
                    tracing::info!(model_id, ?verdict, "B5: smoke probe passed");
                }
                SmokeVerdict::BrokenPunctLoop {
                    dominant_piece,
                    distinct_ids,
                } => {
                    tracing::error!(
                        model_id,
                        dominant_piece,
                        distinct_ids,
                        "B5: smoke probe FAILED — BrokenPunctLoop; refusing to serve"
                    );
                    return Err(format!(
                        "smoke probe failed for '{model_id}': broken_punct_loop \
                         (dominant='{dominant_piece}', distinct_ids={distinct_ids})"
                    ));
                }
                SmokeVerdict::BrokenNan { at_step } => {
                    tracing::error!(
                        model_id,
                        at_step,
                        "B5: smoke probe FAILED — BrokenNan; refusing to serve"
                    );
                    return Err(format!(
                        "smoke probe failed for '{model_id}': broken_nan at step {at_step}"
                    ));
                }
            }
        }

        // Load the requested model.
        tracing::info!(model_id, "slots: loading model");
        let gen = (self.loader)(&entry.abs_path, model_id)
            .map_err(|e| format!("failed to load model '{model_id}': {e}"))?;
        let gen: Arc<dyn Generator> = Arc::from(gen);
        let now = Instant::now();
        // A2: snapshot the effective max ctx once at load time.
        let effective_max_ctx = gen.effective_max_ctx();
        // Fresh decode-lease counter + empty unload-handle slot.
        let decode_lease: DecodeLease = Arc::new(AtomicUsize::new(0));
        let unload_handle: Arc<PLMutex<Option<JoinHandle<()>>>> = Arc::new(PLMutex::new(None));
        let keep_alive = self.idle_policy;
        let resident = {
            // Re-acquire the slots write lock to push the new entry.
            let mut slots = self.slots.write();
            slots.push(LoadedModel {
                id: model_id.to_owned(),
                model: Arc::clone(&gen),
                loaded_at: now,
                last_used: now,
                effective_max_ctx,
                decode_lease: Arc::clone(&decode_lease),
                unload_handle: Arc::clone(&unload_handle),
                keep_alive,
            });
            slots.len()
        };
        tracing::info!(model_id, effective_max_ctx, resident, "slots: model loaded");
        self.arm_or_reset_timer(
            model_id,
            keep_alive,
            &decode_lease,
            &unload_handle,
            /*reset=*/ false,
        );
        Ok((gen, true))
    }

    /// Arm (or reset) the per-model unload timer.
    ///
    /// Cancels any previously armed `JoinHandle` for this slot and spawns a
    /// fresh `sleep(ttl).then(unload)` on the current tokio runtime. When
    /// `policy` is `Pin`, no timer is armed (and any prior timer is cleared).
    ///
    /// Safe to call from sync contexts that are themselves running inside a
    /// tokio runtime (request handlers, the `spawn_blocking` eager preload).
    /// Outside any runtime (unit tests that construct `AppState` directly
    /// without `block_on`), this becomes a no-op so tests don't crash —
    /// real production always runs inside `rt.block_on`.
    ///
    /// H2 contract: the caller controls whether the slot's
    /// `decode_lease` Arc is swapped before this call. The swap MUST happen
    /// at request-triggered reset sites (`ensure_loaded` warm branch and
    /// `reset_keep_alive`) so the H1 identity check in the spawned task can
    /// also detect warm-reset races. The swap MUST NOT happen at the
    /// internal busy re-arm tail (this function's own re-entry after the
    /// busy check) — that path inherits the slot's current lease so
    /// in-flight `DecodeLeaseGuard`s remain visible to subsequent busy
    /// checks. See the matching comments at the caller sites.
    pub fn arm_or_reset_timer(
        &self,
        model_id: &str,
        policy: KeepAlivePolicy,
        decode_lease: &DecodeLease,
        handle_slot: &Arc<PLMutex<Option<JoinHandle<()>>>>,
        reset: bool,
    ) {
        // Hold the handle_slot lock across the entire cancel +
        // spawn + store sequence so concurrent arm calls cannot lose track
        // of in-flight timer handles (the original two-phase implementation
        // had a window where two arms could overwrite each other's handle
        // without aborting the first).
        let mut slot_guard = handle_slot.lock();
        if let Some(prev) = slot_guard.take() {
            prev.abort();
        }
        let Some(ttl) = policy.ttl() else {
            // Pin policy — no timer.
            tracing::info!(
                model_id,
                policy = "pin",
                "keep_alive_armed: pin (no unload timer)"
            );
            return;
        };

        // Verify we're inside a tokio runtime before spawning. If not (unit
        // tests), skip arming — the production server always runs inside
        // `rt.block_on`.
        let Ok(rt_handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                model_id,
                "keep_alive: no tokio runtime — skipping timer arm (likely a unit test)"
            );
            return;
        };

        let model_id_owned = model_id.to_owned();
        let state_clone = self.clone();
        let lease_clone = Arc::clone(decode_lease);
        let handle_slot_clone = Arc::clone(handle_slot);
        let ttl_secs = policy.ttl_secs_for_log();
        // H1+H2: identity token captured at spawn time. After
        // `sleep().await` returns we must verify the slot we were spawned
        // for is still the same slot — otherwise a freshly-reset model
        // would be torn down by a stale TTL fire that won its abort race
        // because the spawned task had already passed its only await
        // point.
        //
        // The identity check catches BOTH races:
        //   (H1) unload-then-reload — the new `LoadedModel` is constructed
        //        with a fresh `Arc<AtomicUsize>` in the cold-load path of
        //        `ensure_loaded`, so the slot's lease pointer differs.
        //   (H2) warm-reset — every `arm_or_reset_timer(reset=true)` call
        //        is paired with a swap of the slot's `decode_lease` Arc at
        //        the caller site (`ensure_loaded` warm branch + the busy
        //        re-arm tail + `reset_keep_alive`), so the slot's lease
        //        pointer differs after a request reset even when the same
        //        `LoadedModel` continues to occupy the slot.
        let captured_lease_ptr = Arc::as_ptr(&lease_clone) as usize;

        if reset {
            tracing::debug!(model_id, ttl_secs, "keep_alive_reset");
        } else {
            tracing::info!(model_id, ttl_secs, "keep_alive_armed");
        }

        // NOT cancel-safe: this future has exactly one await point
        // (`tokio::time::sleep(ttl).await`). If a concurrent arm calls
        // `prev.abort()` while the task is parked on the sleep, the task
        // unwinds and never reaches the unload logic — that is the intended
        // cancel path. But if the abort lands AFTER the sleep has resolved,
        // the task continues to completion and `prev.abort()` is a no-op.
        // The H1 identity check below guards that exact window: we re-check
        // under the write lock that the slot we were spawned for has not been
        // reset (Arc pointer of decode_lease unchanged) before tearing down.
        let jh = rt_handle.spawn(async move {
            tokio::time::sleep(ttl).await;
            // Decode-lease check: if a generation is in flight, do not unload.
            // Re-arm for another TTL period; the next request reset will
            // replace this timer if the model is still being used.
            let busy = lease_clone.load(std::sync::atomic::Ordering::Acquire) > 0;
            if busy {
                tracing::debug!(
                    model_id = %model_id_owned,
                    "keep_alive: decode in flight — deferring unload"
                );
                // Re-arm. Look the model up again under a read lock so a
                // concurrent unload (LRU evict, explicit unload) is observed.
                let next = {
                    let slots = state_clone.slots.read();
                    slots.iter().find(|m| m.id == model_id_owned).map(|loaded| {
                        (
                            loaded.keep_alive,
                            Arc::clone(&loaded.decode_lease),
                            Arc::clone(&loaded.unload_handle),
                        )
                    })
                };
                if let Some((policy2, lease2, handle2)) = next {
                    state_clone.arm_or_reset_timer(
                        &model_id_owned,
                        policy2,
                        &lease2,
                        &handle2,
                        /*reset=*/ true,
                    );
                }
                // Do NOT clear handle_slot_clone here — `arm_or_reset_timer`
                // above stored the NEW timer's JoinHandle in the same slot
                // (handle2 == this task's handle_slot_clone for the resident
                // model), and clearing would orphan it. The re-arm path is
                // self-managing: the next request's reset (or another TTL
                // fire) will overwrite as needed.
                return;
            }
            // H1: identity-check + remove under a single write lock.
            // The captured `lease_ptr` is the Arc::as_ptr of this task's
            // decode_lease; the resident slot's lease pointer must match for
            // this TTL fire to be authoritative. If a request reset the timer
            // (and any nested reset-with-Pin / new-load swap) after this task
            // passed its sleep boundary, the slot's identity has changed and
            // we must NOT unload.
            let evicted = {
                let mut slots_w = state_clone.slots.write();
                let still_authoritative = slots_w
                    .iter()
                    .find(|m| m.id == model_id_owned)
                    .is_some_and(|m| Arc::as_ptr(&m.decode_lease) as usize == captured_lease_ptr);
                if still_authoritative {
                    slots_w
                        .iter()
                        .position(|m| m.id == model_id_owned)
                        .map(|idx| slots_w.swap_remove(idx))
                } else {
                    tracing::debug!(
                        model_id = %model_id_owned,
                        "keep_alive: stale TTL fire — slot was reset"
                    );
                    None
                }
            };
            if let Some(loaded) = evicted {
                state_clone.finalize_unload(&loaded, UnloadReason::IdleTtl(ttl_secs), None);
                // Explicit drop releases the Arc<dyn Generator> and frees
                // the model's GPU residency for the next load.
                drop(loaded);
            }
            // The reference to `handle_slot_clone` is held only to keep the
            // Arc alive until the task ends — no explicit clear here so a
            // concurrent re-arm via reset_keep_alive can't be clobbered.
            drop(handle_slot_clone);
        });
        *slot_guard = Some(jh);
    }

    /// Convenience helper — reset the keep-alive timer for `model_id`
    /// if it is resident, optionally overriding the policy.
    ///
    /// `override_policy = None` keeps the slot's stored policy and just resets
    /// the countdown. `Some(p)` rewrites the slot's policy first (used by the
    /// per-request `keep_alive` body field on native routes).
    pub fn reset_keep_alive(&self, model_id: &str, override_policy: Option<KeepAlivePolicy>) {
        let (policy, lease, handle_slot) = {
            let mut slots = self.slots.write();
            let Some(loaded) = slots.iter_mut().find(|m| m.id == model_id) else {
                return;
            };
            if let Some(p) = override_policy {
                loaded.keep_alive = p;
            }
            // Swap the slot's decode_lease so the previous
            // timer's identity token no longer matches — see the matching
            // swap in `ensure_loaded` warm-reset branch for full rationale.
            // This path is reached from `/v1/models/{id}/load` with a body
            // `keep_alive` override; the caller has not acquired any decode
            // lease guard yet (that endpoint does not decode).
            loaded.decode_lease = Arc::new(AtomicUsize::new(0));
            (
                loaded.keep_alive,
                Arc::clone(&loaded.decode_lease),
                Arc::clone(&loaded.unload_handle),
            )
        };
        self.arm_or_reset_timer(model_id, policy, &lease, &handle_slot, /*reset=*/ true);
    }

    /// Acquire a decode lease for the resident model `model_id`.
    ///
    /// Returns `None` if the model is not resident at lookup time — caller
    /// should already have ensured the model is loaded; this is a safety net
    /// against the cold-start race between `ensure_loaded` and the very next
    /// statement.
    ///
    /// H2: the prior "benign churn" race (timer fire between
    /// `ensure_loaded` and this call evicts the slot) is closed by the
    /// lease-swap at `ensure_loaded`'s warm branch. After a warm reset, the
    /// slot's lease pointer differs from the previous timer's captured
    /// pointer, and the identity check in `arm_or_reset_timer`'s spawned
    /// task bails before evicting. The fresh timer's full TTL window
    /// brackets this lookup — a stale fire inside the window would have
    /// failed the identity check and not unloaded the slot.
    pub fn decode_lease_guard(&self, model_id: &str) -> Option<DecodeLeaseGuard> {
        let lease = self
            .slots
            .read()
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| Arc::clone(&m.decode_lease))?;
        Some(DecodeLeaseGuard::acquire(lease))
    }

    /// Return `(model_path, tokenizer_path)` if both are configured.
    ///
    /// Returns `None` if either path is absent — the audio handler returns 503 in that case.
    pub fn audio_paths(&self) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        match (&self.whisper_model_path, &self.whisper_tokenizer_path) {
            (Some(m), Some(t)) => Some((m.clone(), t.clone())),
            _ => None,
        }
    }

    /// Unload the model `id` if it is currently resident.
    ///
    /// Also evicts all session-cache entries for the model so stale KV
    /// snapshots don't consume slot reservation headroom after unload.
    ///
    /// Returns `true` if the model was unloaded, `false` if it was not loaded.
    pub fn unload(&self, model_id: &str) -> bool {
        self.unload_with_reason(model_id, UnloadReason::Explicit)
    }

    /// Idempotent unload with attribution.
    ///
    /// Cancels any pending unload timer for this slot, removes the slot
    /// (and clears the session cache), and is safe to call concurrently with
    /// the TTL timer / LRU evict — both eventually converge on `slots`
    /// being empty.
    pub fn unload_with_reason(&self, model_id: &str, reason: UnloadReason) -> bool {
        let mut slots = self.slots.write();
        let mut taken: Option<LoadedModel> = None;
        // swap_remove the first matching slot. retain() loses the entry, so
        // we walk by index.
        if let Some(idx) = slots.iter().position(|m| m.id == model_id) {
            taken = Some(slots.swap_remove(idx));
        }
        drop(slots);
        let Some(loaded) = taken else {
            return false;
        };
        self.finalize_unload(&loaded, reason, None);
        // Explicit drop releases the model's Arc<dyn Generator> and frees
        // any associated GPU residency before this call returns.
        drop(loaded);
        true
    }

    /// Shared finaliser for an already-removed slot.
    ///
    /// Single tracing site per unload reason — keeps `model_unload_*` event
    /// emission centralised. Aborts the slot's pending unload timer (if any
    /// — TTL timers that called this themselves have already finished and
    /// the abort is a no-op on the completed handle) and clears the
    /// session-cache entries for the model.
    ///
    /// `requested` is the id of the model whose load triggered a cooperative
    /// evict; ignored for other reasons.
    fn finalize_unload(&self, loaded: &LoadedModel, reason: UnloadReason, requested: Option<&str>) {
        if let Some(h) = loaded.unload_handle.lock().take() {
            h.abort();
        }
        self.session_cache.lock().remove_model(&loaded.id);
        match reason {
            UnloadReason::Explicit => {
                tracing::info!(
                    model_id = %loaded.id,
                    reason = "explicit",
                    "slots: unloading model"
                );
            }
            UnloadReason::IdleTtl(secs) => {
                tracing::debug!(
                    model_id = %loaded.id,
                    idle_secs = secs,
                    "slots: unloaded by idle TTL"
                );
                tracing::info!(
                    model_id = %loaded.id,
                    idle_secs = secs,
                    "model_unload_idle"
                );
            }
            UnloadReason::CooperativeEvict => {
                tracing::info!(
                    model_id = %loaded.id,
                    requested = requested.unwrap_or(""),
                    reason = "cooperative_evict",
                    "model_unload_evict"
                );
            }
        }
    }
}

/// Reason attribution for an unload — drives the tracing event and
/// `model_unload_*` labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "internal — adding a variant requires updating event sites"
)]
pub enum UnloadReason {
    /// Caller invoked `/v1/models/{id}/unload` or `state.unload()` directly.
    Explicit,
    /// TTL timer fired with no active decode lease.
    IdleTtl(u64),
    /// Same-process cooperative evict triggered by loading a new model.
    ///
    /// Covers all current eviction paths — at `max_loaded_models == 1` the
    /// LRU branch in `ensure_loaded` evicts the single resident slot when
    /// a different model is requested, and that path already constructs
    /// `CooperativeEvict`. A dedicated `LruEvict` variant can be added in
    /// the future commit that introduces a direct-LRU path.
    CooperativeEvict,
}
