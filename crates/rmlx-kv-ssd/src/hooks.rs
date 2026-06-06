//! Process-global SSD-tier hook globals (event recorder + Prometheus closures).
//!
//! Migrated from `rmlx_models::kv_cache::mod`. A single MLX process serves
//! one model at a time; one recorder + one closure-set suffices, set once at
//! serve startup before any model load (which is where spill threads spawn).
//!
//! Reads are cheap (`OnceLock` is lock-free after initialisation).

use std::sync::{Arc, OnceLock};

/// Process-wide SSD-tier event recorder (SQLite events table).
///
/// Static visibility kept tight: only the [`set_ssd_event_recorder`] /
/// [`ssd_event_recorder`] accessors below touch it.
static SSD_EVENT_RECORDER: OnceLock<Arc<rmlx_metrics::events::EventRecorder>> = OnceLock::new();

/// Set the process-wide SSD-tier event recorder.
///
/// Must be called before the first spill or hydrate for events to be captured.
/// Subsequent calls are no-ops (first writer wins). Called by the server layer
/// (`rmlx-server`) at startup, before any model load.
pub fn set_ssd_event_recorder(recorder: Arc<rmlx_metrics::events::EventRecorder>) {
    let _ = SSD_EVENT_RECORDER.set(recorder);
}

/// Snapshot the process-global recorder; returns `None` if not yet set.
pub fn ssd_event_recorder() -> Option<Arc<rmlx_metrics::events::EventRecorder>> {
    SSD_EVENT_RECORDER.get().cloned()
}

// ── Process-global Prometheus observation hooks ───────────────────────────────
//
// After a spill or hydrate event is written to SQLite via the EventRecorder, the
// same timing data must also reach the in-process Prometheus accumulators
// (histogram + gauge state in `rmlx-server::openai`). Because `rmlx-kv-ssd`
// cannot depend on `rmlx-server`, the server layer installs lightweight closures
// here at startup and the hot path calls them via these OnceLocks.
//
// Signature: `fn(dur_us: u64, bytes: u64)` — matches the two fields that drive
// the Prometheus histogram observations.

/// Type alias for the `(dur_us, bytes)` Prometheus hook closures.
///
/// Used by [`set_ssd_spill_prom_hook`] and [`set_ssd_hydrate_prom_hook`].
/// Callers in `rmlx-server` may use this alias instead of spelling out the
/// full `Arc<dyn Fn(u64, u64) + Send + Sync>` form.
pub type PromHook = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Type alias for the `(namespace, value)` Prometheus hook closures.
///
/// Used by [`set_ssd_bytes_used_hook`] and [`set_ssd_evict_total_hook`].
/// Callers in `rmlx-server` may use this alias instead of spelling out the
/// full `Arc<dyn Fn(&str, u64) + Send + Sync>` form.
pub type BytesUsedHook = Arc<dyn Fn(&str, u64) + Send + Sync>;

static SSD_SPILL_PROM_HOOK: OnceLock<PromHook> = OnceLock::new();
static SSD_HYDRATE_PROM_HOOK: OnceLock<PromHook> = OnceLock::new();
/// Hook that receives `(namespace, bytes)` after startup maintenance and after
/// each spill, so the Prometheus `rmlx_ssd_bytes_used` gauge stays current.
static SSD_BYTES_USED_HOOK: OnceLock<BytesUsedHook> = OnceLock::new();
/// Hook that receives `(namespace, evicted_count)` after startup eviction,
/// so the Prometheus `rmlx_ssd_evict_total` counter is updated.
static SSD_EVICT_TOTAL_HOOK: OnceLock<BytesUsedHook> = OnceLock::new();

/// Register the Prometheus observation hook for SSD spill events.
///
/// The closure receives `(dur_us, bytes)` immediately after each spill completes.
/// Called by `rmlx-server` at startup; subsequent calls are no-ops (first wins).
pub fn set_ssd_spill_prom_hook(hook: PromHook) {
    let _ = SSD_SPILL_PROM_HOOK.set(hook);
}

/// Register the Prometheus observation hook for SSD hydrate events.
///
/// The closure receives `(dur_us, bytes)` immediately after each hydrate completes.
/// Called by `rmlx-server` at startup; subsequent calls are no-ops (first wins).
pub fn set_ssd_hydrate_prom_hook(hook: PromHook) {
    let _ = SSD_HYDRATE_PROM_HOOK.set(hook);
}

/// Register the Prometheus gauge hook for the on-disk SSD byte count.
///
/// The closure receives `(namespace, bytes_used)` after startup maintenance
/// (when the index is first opened) and after each spill updates the footprint.
/// Called by `rmlx-server` at startup; subsequent calls are no-ops (first wins).
pub fn set_ssd_bytes_used_hook(hook: BytesUsedHook) {
    let _ = SSD_BYTES_USED_HOOK.set(hook);
}

/// Register the Prometheus counter hook for startup LRU eviction count.
///
/// The closure receives `(namespace, evicted_count)` after startup
/// `evict_lru_until` completes. `evicted_count` is the number of blocks
/// removed; 0 when the index was already within budget.
/// Called by `rmlx-server` at startup; subsequent calls are no-ops.
pub fn set_ssd_evict_total_hook(hook: BytesUsedHook) {
    let _ = SSD_EVICT_TOTAL_HOOK.set(hook);
}

/// Call the registered spill Prometheus hook, if any. No-op when unset.
pub fn call_ssd_spill_prom_hook(dur_us: u64, bytes: u64) {
    if let Some(hook) = SSD_SPILL_PROM_HOOK.get() {
        hook(dur_us, bytes);
    }
}

/// Call the registered hydrate Prometheus hook, if any. No-op when unset.
pub fn call_ssd_hydrate_prom_hook(dur_us: u64, bytes: u64) {
    if let Some(hook) = SSD_HYDRATE_PROM_HOOK.get() {
        hook(dur_us, bytes);
    }
}

/// Call the registered bytes-used Prometheus gauge hook, if any. No-op when unset.
pub fn call_ssd_bytes_used_hook(namespace: &str, bytes: u64) {
    if let Some(hook) = SSD_BYTES_USED_HOOK.get() {
        hook(namespace, bytes);
    }
}

/// Call the registered evict-total counter hook, if any. No-op when unset.
pub fn call_ssd_evict_total_hook(namespace: &str, evicted: u64) {
    if let Some(hook) = SSD_EVICT_TOTAL_HOOK.get() {
        hook(namespace, evicted);
    }
}
