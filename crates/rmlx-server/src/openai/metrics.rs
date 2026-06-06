//! Metrics snapshot gathering and HTTP endpoints:
//! - `GET /metrics/cache` (N19) — JSON
//! - `GET /metrics` (F5) — Prometheus text exposition
//! - `GET /v1/metrics` — rolling request-level JSON summary

use std::fmt::Write as _;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::engine::Generator;

use super::state::{snapshot_ssd_tier, HIST_BUCKETS_US};
use super::state::{ApiErrorCategory, AppState, ItlSample, TtftSample};

// ── Metrics snapshot types ────────────────────────────────────────────────────

/// Per-model cache stats snapshot.
pub(crate) struct ModelMetrics {
    pub model_id: String,
    /// Prompt-cache hits (None if the generator has no cache stats).
    pub hits: Option<u64>,
    pub misses: Option<u64>,
    pub evictions: Option<u64>,
    pub cache_bytes: Option<u64>,
    pub block_hits: Option<u64>,
    pub block_misses: Option<u64>,
    pub partial_hits: Option<u64>,
    /// SSD-tier hydrate hits (RAM misses served from the `.kvb` tier).
    pub ssd_hits: Option<u64>,
    /// Last-request KV-cache allocation size (N16).
    pub kv_cache_bytes: u64,
    /// Metal allocator peak (C7).
    pub metal_peak_alloc_bytes: Option<u64>,
}

/// Process-wide metrics snapshot gathered atomically (no cross-snapshot drift).
///
/// Both `/metrics/cache` (JSON) and `/metrics` (Prometheus text) are built
/// from this struct so they always reflect the same underlying values.
pub(crate) struct MetricsSnapshot {
    /// Per-model prompt-cache / kv-cache stats.
    pub models: Vec<ModelMetrics>,
    /// Raw TTFT samples from the ring (oldest first). Used to derive
    /// per-endpoint percentiles for Prometheus and the raw array for JSON.
    pub ttft_samples: Vec<TtftSample>,
    /// ITL aggregate samples from the ring (oldest first). Each entry
    /// already carries pre-computed `p50_ms`, `p95_ms`, `mean_ms`.
    pub itl_samples: Vec<ItlSample>,
    /// F14: lifetime prompt-token counter.
    pub tokens_in: u64,
    /// F14: lifetime completion-token counter.
    pub tokens_out: u64,
    /// F8: per-category error counts keyed by `ApiErrorCategory::as_str()`.
    pub error_counts: Vec<(&'static str, u64)>,
    /// J4 process memory (None if the kernel call failed).
    pub proc_mem: Option<rmlx_core::mach_mem::ProcMem>,
    /// server uptime in fractional seconds at snapshot time.
    pub uptime_s: f64,
    /// current in-flight request count (gpu_pending Relaxed load).
    pub in_flight: usize,
    /// lifetime count of admitted requests.
    pub requests_started: u64,
    /// lifetime count of requests that completed successfully.
    pub requests_completed: u64,
    /// lifetime count of requests that returned an engine error.
    pub requests_failed: u64,
    /// mean decode throughput in tokens/s, derived from the ITL ring.
    ///
    /// Formula: for each `ItlSample` with `step_count >= 2`, compute
    /// `1000.0 / mean_ms` (tokens per second from inter-token latency), then
    /// take the mean across all samples in the ring. `None` when the ITL ring
    /// is empty (no completed multi-token requests yet).
    pub avg_decode_tok_s: Option<f64>,
}

/// Gather a consistent snapshot from `AppState`.
///
/// Holds each lock only for the minimum duration needed to clone the data,
/// then releases before accessing the next field. No lock is held across the
/// generator trait calls.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(crate) fn gather_metrics(state: &AppState) -> MetricsSnapshot {
    // Collect (id, Arc) under the read-lock, release before generator calls.
    let slot_info: Vec<(String, Arc<dyn Generator>)> = {
        let slots = state.slots.read();
        slots
            .iter()
            .map(|m| (m.id.clone(), Arc::clone(&m.model)))
            .collect()
    };

    let models: Vec<ModelMetrics> = slot_info
        .into_iter()
        .filter_map(|(model_id, generator)| {
            let stats_opt = generator.cache_stats();
            let kv_cache_bytes = generator.kv_cache_bytes();
            let metal_peak_alloc_bytes = rmlx_mlx::mlx_peak_memory_bytes();

            // Skip entries that carry nothing useful (no stats and no kv bytes).
            if stats_opt.is_none() && kv_cache_bytes == 0 && metal_peak_alloc_bytes.is_none() {
                return None;
            }

            Some(match stats_opt {
                None => ModelMetrics {
                    model_id,
                    hits: None,
                    misses: None,
                    evictions: None,
                    cache_bytes: None,
                    block_hits: None,
                    block_misses: None,
                    partial_hits: None,
                    ssd_hits: None,
                    kv_cache_bytes,
                    metal_peak_alloc_bytes,
                },
                Some(stats) => {
                    tracing::debug!(
                        model_id = %model_id,
                        hits = stats.hits,
                        misses = stats.misses,
                        evictions = stats.evictions,
                        bytes = stats.bytes,
                        kv_cache_bytes,
                        "gather_metrics: prompt-cache (N19) + kv-cache (N16)"
                    );
                    ModelMetrics {
                        model_id,
                        hits: Some(stats.hits),
                        misses: Some(stats.misses),
                        evictions: Some(stats.evictions),
                        cache_bytes: Some(stats.bytes),
                        block_hits: Some(stats.block_hits),
                        block_misses: Some(stats.block_misses),
                        partial_hits: Some(stats.partial_hits),
                        ssd_hits: Some(stats.ssd_hits),
                        kv_cache_bytes,
                        metal_peak_alloc_bytes,
                    }
                }
            })
        })
        .collect();

    // L6: snapshot TTFT ring (brief lock).
    let ttft_samples: Vec<TtftSample> = {
        let ring = state.ttft_store.lock();
        ring.iter().cloned().collect()
    };

    // M30: snapshot ITL ring (brief lock).
    let itl_samples: Vec<ItlSample> = {
        let ring = state.itl_store.lock();
        ring.iter().cloned().collect()
    };

    // F14.
    let tokens_in = state.tokens_in.load(Relaxed);
    let tokens_out = state.tokens_out.load(Relaxed);

    // F8.
    let error_counts = {
        use ApiErrorCategory::{
            AdmissionSla503, BadRequest, ContextOverflow, Internal, NotFound, OomKvCache, OomLoad,
            OomMidStream, RateLimit, Timeout, Upstream,
        };
        let cats = [
            BadRequest,
            ContextOverflow,
            NotFound,
            OomLoad,
            OomKvCache,
            OomMidStream,
            Timeout,
            Upstream,
            Internal,
            RateLimit,
            AdmissionSla503,
        ];
        cats.iter()
            .map(|&cat| {
                let n = match cat {
                    BadRequest => state.error_counts.bad_request.load(Relaxed),
                    ContextOverflow => state.error_counts.context_overflow.load(Relaxed),
                    NotFound => state.error_counts.not_found.load(Relaxed),
                    OomLoad => state.error_counts.oom_load.load(Relaxed),
                    OomKvCache => state.error_counts.oom_kv_cache.load(Relaxed),
                    OomMidStream => state.error_counts.oom_mid_stream.load(Relaxed),
                    Timeout => state.error_counts.timeout.load(Relaxed),
                    Upstream => state.error_counts.upstream.load(Relaxed),
                    Internal => state.error_counts.internal.load(Relaxed),
                    RateLimit => state.error_counts.rate_limit.load(Relaxed),
                    AdmissionSla503 => state.error_counts.admission_sla_503.load(Relaxed),
                };
                (cat.as_str(), n)
            })
            .collect()
    };

    let proc_mem = rmlx_core::mach_mem::read_proc_mem().ok();

    // uptime + in-flight + request counters.
    let uptime_s = state.started_at.elapsed().as_secs_f64();
    let in_flight = state.gpu_pending.load(Relaxed);
    let requests_started = state.requests_started.load(Relaxed);
    let requests_completed = state.requests_completed.load(Relaxed);
    let requests_failed = state.requests_failed.load(Relaxed);

    // avg decode throughput from ITL ring.
    // Formula: 1000.0 / mean_ms per sample (ITL mean → tok/s), averaged over
    // all ring entries that have at least 2 decode steps (step_count >= 2 means
    // at least one inter-token interval was recorded).
    let avg_decode_tok_s = {
        let valid: Vec<f64> = itl_samples
            .iter()
            .filter(|s| s.step_count >= 2 && s.mean_ms > 0.0)
            .map(|s| 1000.0 / s.mean_ms)
            .collect();
        if valid.is_empty() {
            None
        } else {
            Some(valid.iter().sum::<f64>() / valid.len() as f64)
        }
    };

    MetricsSnapshot {
        models,
        ttft_samples,
        itl_samples,
        tokens_in,
        tokens_out,
        error_counts,
        proc_mem,
        uptime_s,
        in_flight,
        requests_started,
        requests_completed,
        requests_failed,
        avg_decode_tok_s,
    }
}

// ── Route: GET /metrics/cache (N19) ──────────────────────────────────────────

/// Prompt-cache hit/miss/bytes stats + TTFT samples + load-phase spans for
/// the currently-loaded model.
///
/// Acquires the slot lock only long enough to clone the generator `Arc` and
/// the model id — the actual metric reads are outside the lock.
/// JSON shape:
/// ```json
/// { "models": [{ "model_id": "...", "hits": 0, "misses": 0,
/// "bytes": 0, "hit_rate": 0.0, "kv_cache_bytes": 0,
/// "load_phases": { "mmap_ms": 0, "dequant_ms": 0,
/// "gpu_residency_ms": 0, "first_kernel_ready_ms": 0,
/// "total_load_ms": 0 } }],
/// "ttft": [{ "model_id": "...", "ttft_ms": 123 }, ...] }
/// ```
/// Returns an empty `models` array when no model is loaded or the cache has
/// not been used yet. `ttft` contains the last N per-request TTFT samples
/// across all models in chronological order (oldest first).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(crate) async fn metrics_cache(State(state): State<AppState>) -> Response {
    let snap = gather_metrics(&state);

    // Rebuild per-model JSON from the snapshot; load_phases must still come
    // from the generator directly (not in the snapshot — low-traffic data).
    let slot_arcs: Vec<(String, Arc<dyn Generator>)> = {
        let slots = state.slots.read();
        slots
            .iter()
            .map(|m| (m.id.clone(), Arc::clone(&m.model)))
            .collect()
    };

    let models_json: Vec<Value> = snap
        .models
        .iter()
        .map(|m| {
            // Retrieve load-phase timing from the live generator if available.
            let load_phases_json = slot_arcs
                .iter()
                .find(|(id, _)| *id == m.model_id)
                .and_then(|(_, gen)| gen.load_phases())
                .map(|p| {
                    serde_json::json!({
                        "mmap_ms": p.mmap_ms,
                        "dequant_ms": p.dequant_ms,
                        "gpu_residency_ms": p.gpu_residency_ms,
                        "first_kernel_ready_ms": p.first_kernel_ready_ms,
                        "total_load_ms": p.total_load_ms,
                    })
                });

            let mut entry = serde_json::json!({ "model_id": m.model_id });

            if let (Some(hits), Some(misses), Some(evictions), Some(bytes)) =
                (m.hits, m.misses, m.evictions, m.cache_bytes)
            {
                let total = hits + misses;
                let hit_rate = if total == 0 {
                    0.0_f64
                } else {
                    hits as f64 / total as f64
                };
                let partial_hit_rate = if hits == 0 {
                    0.0_f64
                } else {
                    m.partial_hits.unwrap_or(0) as f64 / hits as f64
                };
                entry["hits"] = serde_json::json!(hits);
                entry["misses"] = serde_json::json!(misses);
                entry["evictions"] = serde_json::json!(evictions);
                entry["bytes"] = serde_json::json!(bytes);
                entry["hit_rate"] = serde_json::json!(hit_rate);
                entry["kv_cache_bytes"] = serde_json::json!(m.kv_cache_bytes);
                entry["metal_peak_alloc_bytes"] = serde_json::json!(m.metal_peak_alloc_bytes);
                entry["block_hits"] = serde_json::json!(m.block_hits);
                entry["block_misses"] = serde_json::json!(m.block_misses);
                entry["partial_hits"] = serde_json::json!(m.partial_hits);
                entry["partial_hit_rate"] = serde_json::json!(partial_hit_rate);
                entry["ssd_hits"] = serde_json::json!(m.ssd_hits);
            } else {
                if m.kv_cache_bytes > 0 {
                    entry["kv_cache_bytes"] = serde_json::json!(m.kv_cache_bytes);
                }
                if let Some(peak) = m.metal_peak_alloc_bytes {
                    entry["metal_peak_alloc_bytes"] = serde_json::json!(peak);
                }
            }

            if let Some(lp) = load_phases_json {
                entry["load_phases"] = lp;
            }
            entry
        })
        .collect();

    let ttft_json: Vec<Value> = snap
        .ttft_samples
        .iter()
        .map(|s| serde_json::json!({ "model_id": s.model_id, "ttft_ms": s.ttft_ms }))
        .collect();

    let itl_json: Vec<Value> = snap
        .itl_samples
        .iter()
        .map(|s| {
            serde_json::json!({
                "model_id": s.model_id,
                "p50_ms": s.p50_ms,
                "p95_ms": s.p95_ms,
                "step_mean_ms": s.mean_ms,
                "step_count": s.step_count,
            })
        })
        .collect();

    let error_counts_json: Value = snap
        .error_counts
        .iter()
        .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
        .collect::<serde_json::Map<_, _>>()
        .into();

    let body = serde_json::json!({
        "models": models_json,
        "ttft": ttft_json,
        "itl": itl_json,
        "tokens_in": snap.tokens_in,
        "tokens_out": snap.tokens_out,
        "error_counts": error_counts_json,
    });
    (StatusCode::OK, Json(body)).into_response()
}

// ── Route: GET /metrics (F5) — Prometheus text exposition ────────────────────

/// Prometheus text-format exposition of the same metrics as `/metrics/cache`.
///
/// Format: Prometheus text exposition format v0.0.4.
/// Content-Type: `text/plain; version=0.0.4`.
///
/// Every numeric line is preceded by `# HELP` and `# TYPE`.
/// Counters (lifetime / cumulative): tokens, error counts, cache hit/miss.
/// Gauges (point-in-time): cache bytes, kv_cache_bytes, TTFT percentiles,
/// ITL percentiles, process RSS / phys_footprint.
///
/// No new dependency is used — the Prometheus text format is hand-rolled.
pub(crate) async fn metrics_prometheus(State(state): State<AppState>) -> Response {
    let snap = gather_metrics(&state);
    let text = render_prometheus(&snap);
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        text,
    )
        .into_response()
}

// ── Route: GET /v1/metrics — rolling request-level JSON summary ──────

/// Rolling request-level JSON summary at `GET /v1/metrics`.
///
/// Mirrors the mlx-vlm `ServerMetricsStore.snapshot()` key names for
/// cross-backend comparability (reads this; the bench pipeline too).
///
/// JSON shape:
/// ```json
/// {
/// "uptime_s": 12.3,
/// "in_flight": 0,
/// "requests_started": 5,
/// "requests_completed": 4,
/// "requests_failed": 0,
/// "avg_request_tok_s": null,
/// "avg_decode_tok_s": null,
/// "tokens_in": 1234,
/// "tokens_out": 567,
/// "last_error": null
/// }
/// ```
///
/// `avg_decode_tok_s` is now populated from the ITL ring
/// (formula: mean of `1000 / mean_itl_ms` across ring samples with ≥2 steps).
/// `avg_request_tok_s` remains `null` — total per-request wall time is not
/// tracked (only TTFT and ITL are; TTFT alone is not enough to derive tok/s).
/// `last_error` is reserved for future use.
pub(crate) async fn metrics_v1_summary(State(state): State<AppState>) -> Json<Value> {
    // Reuse gather_metrics so avg_decode_tok_s computation is in one place.
    let snap = gather_metrics(&state);

    tracing::debug!(
        uptime_s = snap.uptime_s,
        in_flight = snap.in_flight,
        requests_started = snap.requests_started,
        requests_completed = snap.requests_completed,
        requests_failed = snap.requests_failed,
        tokens_in = snap.tokens_in,
        tokens_out = snap.tokens_out,
        avg_decode_tok_s = ?snap.avg_decode_tok_s,
        "/v1/metrics snapshot"
    );

    Json(json!({
        "uptime_s": snap.uptime_s,
        "in_flight": snap.in_flight,
        "requests_started": snap.requests_started,
        "requests_completed": snap.requests_completed,
        "requests_failed": snap.requests_failed,
        "avg_request_tok_s": null,
        "avg_decode_tok_s": snap.avg_decode_tok_s,
        "tokens_in": snap.tokens_in,
        "tokens_out": snap.tokens_out,
        "last_error": null,
    }))
}

/// Render a `MetricsSnapshot` as Prometheus text exposition format v0.0.4.
///
/// Extracted as a free function so unit tests can call it without an HTTP
/// server or real AppState.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
pub(crate) fn render_prometheus(snap: &MetricsSnapshot) -> String {
    let mut out = String::with_capacity(2048);

    // ── F14: lifetime token counters (counter) ────────────────────────────────
    out.push_str(
        "# HELP rmlx_lifetime_tokens_total Lifetime cumulative token count since server start.\n",
    );
    out.push_str("# TYPE rmlx_lifetime_tokens_total counter\n");
    // write!(String) is infallible — let _ discards the unit Ok.
    let _ = write!(
        out,
        "rmlx_lifetime_tokens_total{{direction=\"in\"}} {}\n",
        snap.tokens_in
    );
    let _ = write!(
        out,
        "rmlx_lifetime_tokens_total{{direction=\"out\"}} {}\n",
        snap.tokens_out
    );

    // ── F8: per-category API error counters (counter) ─────────────────────────
    out.push_str(
        "# HELP rmlx_api_errors_total Lifetime API error count by category since server start.\n",
    );
    out.push_str("# TYPE rmlx_api_errors_total counter\n");
    for (cat, n) in &snap.error_counts {
        let _ = write!(out, "rmlx_api_errors_total{{category=\"{cat}\"}} {n}\n");
    }

    // ── N19: prompt-cache hit / miss (counter) ────────────────────────────────
    out.push_str("# HELP rmlx_prompt_cache_total Lifetime prompt-cache result count.\n");
    out.push_str("# TYPE rmlx_prompt_cache_total counter\n");
    for m in &snap.models {
        if let (Some(hits), Some(misses)) = (m.hits, m.misses) {
            let _ = write!(
                out,
                "rmlx_prompt_cache_total{{model=\"{}\",result=\"hit\"}} {hits}\n",
                m.model_id
            );
            let _ = write!(
                out,
                "rmlx_prompt_cache_total{{model=\"{}\",result=\"miss\"}} {misses}\n",
                m.model_id
            );
        }
    }

    // ── SSD-tier hydrate hits (counter) ───────────────────────────────
    out.push_str(
        "# HELP rmlx_prompt_cache_ssd_hits_total Lifetime RAM-miss-served-from-SSD count.\n",
    );
    out.push_str("# TYPE rmlx_prompt_cache_ssd_hits_total counter\n");
    for m in &snap.models {
        if let Some(ssd_hits) = m.ssd_hits {
            let _ = write!(
                out,
                "rmlx_prompt_cache_ssd_hits_total{{model=\"{}\"}} {ssd_hits}\n",
                m.model_id
            );
        }
    }

    // ── N19: prompt-cache bytes (gauge) ──────────────────────────────────────
    out.push_str("# HELP rmlx_prompt_cache_bytes Current prompt-cache memory usage in bytes.\n");
    out.push_str("# TYPE rmlx_prompt_cache_bytes gauge\n");
    for m in &snap.models {
        if let Some(bytes) = m.cache_bytes {
            let _ = write!(
                out,
                "rmlx_prompt_cache_bytes{{model=\"{}\"}} {bytes}\n",
                m.model_id
            );
        }
    }

    // ── N16: KV-cache bytes (gauge) ───────────────────────────────────────────
    out.push_str("# HELP rmlx_kv_cache_bytes Last-request KV-cache allocation in bytes.\n");
    out.push_str("# TYPE rmlx_kv_cache_bytes gauge\n");
    for m in &snap.models {
        if m.kv_cache_bytes > 0 {
            let _ = write!(
                out,
                "rmlx_kv_cache_bytes{{model=\"{}\"}} {}\n",
                m.model_id, m.kv_cache_bytes
            );
        }
    }

    // ── C7: Metal peak allocator bytes (gauge) ────────────────────────────────
    out.push_str("# HELP rmlx_metal_peak_alloc_bytes Metal allocator peak allocation in bytes.\n");
    out.push_str("# TYPE rmlx_metal_peak_alloc_bytes gauge\n");
    for m in &snap.models {
        if let Some(peak) = m.metal_peak_alloc_bytes {
            let _ = write!(
                out,
                "rmlx_metal_peak_alloc_bytes{{model=\"{}\"}} {peak}\n",
                m.model_id
            );
        }
    }

    // ── L6: TTFT percentiles from ring (gauge) ────────────────────────────────
    // Derive p50/p95/p99 from the raw ring samples.
    if !snap.ttft_samples.is_empty() {
        let mut vals: Vec<u64> = snap.ttft_samples.iter().map(|s| s.ttft_ms).collect();
        vals.sort_unstable();
        let p50 = percentile_u64(&vals, 50);
        let p95 = percentile_u64(&vals, 95);
        let p99 = percentile_u64(&vals, 99);
        out.push_str(
            "# HELP rmlx_ttft_ms Time to first token in milliseconds (from ring buffer).\n",
        );
        out.push_str("# TYPE rmlx_ttft_ms gauge\n");
        let _ = write!(out, "rmlx_ttft_ms{{quantile=\"0.50\"}} {p50}\n");
        let _ = write!(out, "rmlx_ttft_ms{{quantile=\"0.95\"}} {p95}\n");
        let _ = write!(out, "rmlx_ttft_ms{{quantile=\"0.99\"}} {p99}\n");
    }

    // ── M30: ITL percentiles from ring (gauge) ────────────────────────────────
    // Use the last sample's pre-computed percentiles (most recent request).
    if let Some(last) = snap.itl_samples.last() {
        out.push_str("# HELP rmlx_itl_ms Inter-token latency in milliseconds (last request).\n");
        out.push_str("# TYPE rmlx_itl_ms gauge\n");
        let _ = write!(out, "rmlx_itl_ms{{quantile=\"0.50\"}} {}\n", last.p50_ms);
        let _ = write!(out, "rmlx_itl_ms{{quantile=\"0.95\"}} {}\n", last.p95_ms);
    }

    // ── J4: process memory (gauge) ────────────────────────────────────────────
    if let Some(mem) = &snap.proc_mem {
        out.push_str("# HELP rmlx_process_rss_bytes Process resident set size in bytes.\n");
        out.push_str("# TYPE rmlx_process_rss_bytes gauge\n");
        let _ = write!(out, "rmlx_process_rss_bytes {}\n", mem.rss_bytes);

        out.push_str("# HELP rmlx_process_phys_footprint_bytes Process physical memory footprint in bytes (Activity Monitor metric).\n");
        out.push_str("# TYPE rmlx_process_phys_footprint_bytes gauge\n");
        let _ = write!(
            out,
            "rmlx_process_phys_footprint_bytes {}\n",
            mem.phys_footprint_bytes
        );
    }

    // ── server uptime + in-flight gauges ───────────────────────────────
    // These are server-global (not per-model) and always rendered even with
    // zero models loaded — the gauges must appear on every /metrics response.
    out.push_str("# HELP rmlx_uptime_seconds Server uptime in seconds since start.\n");
    out.push_str("# TYPE rmlx_uptime_seconds gauge\n");
    let _ = write!(out, "rmlx_uptime_seconds {}\n", snap.uptime_s);

    out.push_str("# HELP rmlx_in_flight Current number of in-flight (admitted) requests.\n");
    out.push_str("# TYPE rmlx_in_flight gauge\n");
    let _ = write!(out, "rmlx_in_flight {}\n", snap.in_flight);

    // ── avg decode throughput ──────────────────────────────────────────
    if let Some(avg_tok_s) = snap.avg_decode_tok_s {
        out.push_str(
            "# HELP rmlx_avg_decode_tok_s Mean decode throughput in tokens/s (avg of ITL ring, formula: 1000/mean_itl_ms).\n",
        );
        out.push_str("# TYPE rmlx_avg_decode_tok_s gauge\n");
        let _ = write!(out, "rmlx_avg_decode_tok_s {avg_tok_s:.3}\n");
    }

    // ── request lifecycle counters ─────────────────────────────────────
    out.push_str(
        "# HELP rmlx_requests_total Lifetime request count by outcome since server start.\n",
    );
    out.push_str("# TYPE rmlx_requests_total counter\n");
    let _ = write!(
        out,
        "rmlx_requests_total{{outcome=\"started\"}} {}\n",
        snap.requests_started
    );
    let _ = write!(
        out,
        "rmlx_requests_total{{outcome=\"completed\"}} {}\n",
        snap.requests_completed
    );
    let _ = write!(
        out,
        "rmlx_requests_total{{outcome=\"failed\"}} {}\n",
        snap.requests_failed
    );

    // ── SSD-tier observability (step2) ────────────────────────────────────────
    let ssd = snapshot_ssd_tier();

    // ssd_bytes_used gauge (per namespace).
    if !ssd.bytes_used.is_empty() {
        out.push_str(
            "# HELP rmlx_ssd_bytes_used Current on-disk KV-block cache footprint in bytes.\n",
        );
        out.push_str("# TYPE rmlx_ssd_bytes_used gauge\n");
        for (ns, bytes) in &ssd.bytes_used {
            let _ = write!(out, "rmlx_ssd_bytes_used{{namespace=\"{ns}\"}} {bytes}\n");
        }
    }

    // ssd_evict_total counter — emit when the tier is active (bytes_used has
    // entries from startup_maintenance) so the metric is always queryable via
    // Grafana even when no evictions have fired on this instance.
    let tier_active = !ssd.bytes_used.is_empty()
        || !ssd.evict_total.is_empty()
        || ssd.spill.count > 0
        || ssd.hydrate.count > 0;
    if tier_active {
        out.push_str("# HELP rmlx_ssd_evict_total Lifetime SSD-tier LRU eviction count.\n");
        out.push_str("# TYPE rmlx_ssd_evict_total counter\n");
        for (ns, n) in &ssd.evict_total {
            let _ = write!(out, "rmlx_ssd_evict_total{{namespace=\"{ns}\"}} {n}\n");
        }
        // Emit 0-count rows for namespaces known from bytes_used that have no evictions.
        for ns in ssd.bytes_used.keys() {
            if !ssd.evict_total.contains_key(ns) {
                let _ = write!(out, "rmlx_ssd_evict_total{{namespace=\"{ns}\"}} 0\n");
            }
        }
    }

    // ssd_spill_us histogram — always emit HELP/TYPE when the tier is active
    // (bytes_used is non-empty after startup_maintenance), so the metric is
    // queryable even when no spills fired on this instance (hydrate-only runs).
    if tier_active {
        out.push_str("# HELP rmlx_ssd_spill_us Per-spill duration in microseconds.\n");
        out.push_str("# TYPE rmlx_ssd_spill_us histogram\n");
        // H1 fix: use `ssd.spill.count` (not `count + count_inf_overflow`) for
        // both the +Inf bucket and _count. Every call to `observe` increments
        // `count` exactly once, including observations beyond the last finite
        // bucket. `count_inf_overflow` is a diagnostic-only field and must NOT
        // be added here — doing so would count overflow observations twice.
        let total_count = ssd.spill.count;
        for (i, &le) in HIST_BUCKETS_US.iter().enumerate() {
            let _ = write!(
                out,
                "rmlx_ssd_spill_us_bucket{{le=\"{le}\"}} {}\n",
                ssd.spill.buckets[i]
            );
        }
        let _ = write!(
            out,
            "rmlx_ssd_spill_us_bucket{{le=\"+Inf\"}} {total_count}\n"
        );
        let _ = write!(out, "rmlx_ssd_spill_us_sum {}\n", ssd.spill.sum_us);
        let _ = write!(out, "rmlx_ssd_spill_us_count {total_count}\n");
    }

    // ssd_hydrate_us histogram.
    if ssd.hydrate.count > 0 {
        out.push_str("# HELP rmlx_ssd_hydrate_us Per-hydrate duration in microseconds.\n");
        out.push_str("# TYPE rmlx_ssd_hydrate_us histogram\n");
        // H1 fix: same rationale as spill — use `count` directly, not `count +
        // count_inf_overflow`.
        let total_count = ssd.hydrate.count;
        for (i, &le) in HIST_BUCKETS_US.iter().enumerate() {
            let _ = write!(
                out,
                "rmlx_ssd_hydrate_us_bucket{{le=\"{le}\"}} {}\n",
                ssd.hydrate.buckets[i]
            );
        }
        let _ = write!(
            out,
            "rmlx_ssd_hydrate_us_bucket{{le=\"+Inf\"}} {total_count}\n"
        );
        let _ = write!(out, "rmlx_ssd_hydrate_us_sum {}\n", ssd.hydrate.sum_us);
        let _ = write!(out, "rmlx_ssd_hydrate_us_count {total_count}\n");
    }

    // Prometheus spec: text format must end with a newline.
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// Nearest-rank percentile over a sorted slice of `u64`.
///
/// Returns 0 for an empty slice. Uses the "nearest rank" method: index =
/// ceil(p/100 * n) − 1, clamped to `[0, n−1]`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
pub(crate) fn percentile_u64(sorted: &[u64], p: u8) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    // ceil(p/100 * n) - 1, nearest-rank method.
    let rank = (p as usize * n).div_ceil(100).saturating_sub(1);
    sorted[rank.min(n - 1)]
}
