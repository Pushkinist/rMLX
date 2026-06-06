//! Metric registry per `docs/METRICS_DB.md` §4.
//!
//! Maps metric names to `(unit, direction)` pairs. The registry is the
//! authoritative source for what constitutes a valid metric name and how
//! to interpret its direction (`Higher` = better, `Lower` = better).
//! Unregistered metric names are rejected at ingest time.
//!
//! # Public API
//!
//! - [`lookup`] — resolve a metric name to its unit string and [`Direction`].
//! - [`Direction`] — `Higher` or `Lower` (better) enum.
//! - [`Coverage`] — enum indicating whether a backend reports a given metric.
//! - [`coverage`] — check whether a `(backend, metric)` pair has known coverage.

use crate::error::{Error, Result};

// ── Direction ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed enum — exactly two metric directions from the METRICS_DB spec; adding a direction requires updating the registry and all comparison logic"
)]
/// Metric optimization direction per docs/METRICS_DB.md §4.
pub enum Direction {
    /// A larger value is better (e.g. tokens per second).
    HigherBetter,
    /// A smaller value is better (e.g. latency in ms).
    LowerBetter,
}

impl Direction {
    /// Returns the canonical DB string for this direction.
    pub fn as_str(self) -> &'static str {
        match self {
            Direction::HigherBetter => "higher_better",
            Direction::LowerBetter => "lower_better",
        }
    }

    /// Parses `"higher_better"` or `"lower_better"`; returns `Error::UnknownDirection` otherwise.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "higher_better" => Ok(Direction::HigherBetter),
            "lower_better" => Ok(Direction::LowerBetter),
            _ => Err(Error::UnknownDirection(s.to_string())),
        }
    }
}

// ── METRICS const ─────────────────────────────────────────────────────────────

/// Canonical metric name → (unit, direction). Add new metrics here AND in §4.
pub const METRICS: &[(&str, &str, Direction)] = &[
    ("decode_tps_warm", "tps", Direction::HigherBetter),
    ("decode_tps_cold", "tps", Direction::HigherBetter),
    ("prefill_tps", "tps", Direction::HigherBetter),
    ("overall_tps", "tps", Direction::HigherBetter),
    ("ttft_warm_ms", "ms", Direction::LowerBetter),
    ("ttft_cold_ms", "ms", Direction::LowerBetter),
    ("itl_p50_ms", "ms", Direction::LowerBetter),
    ("itl_p95_ms", "ms", Direction::LowerBetter),
    ("step_ms_mean", "ms", Direction::LowerBetter),
    ("model_load_ms", "ms", Direction::LowerBetter),
    ("peak_rss_mb", "mb", Direction::LowerBetter),
    ("metal_peak_alloc_mb", "mb", Direction::LowerBetter),
    ("kv_cache_bytes", "bytes", Direction::LowerBetter),
    ("tps_per_gb_ram", "ratio", Direction::HigherBetter),
    ("task_pass_at_1", "ratio", Direction::HigherBetter),
    // N19: prompt-cache hit/miss/bytes counters.
    ("prompt_cache_hits", "count", Direction::HigherBetter),
    ("prompt_cache_misses", "count", Direction::LowerBetter),
    ("prompt_cache_bytes", "bytes", Direction::LowerBetter),
    // C6: block-level prompt-cache counters (monotonic).
    ("prompt_cache_block_hits", "count", Direction::HigherBetter),
    ("prompt_cache_block_misses", "count", Direction::LowerBetter),
    (
        "prompt_cache_partial_hits",
        "count",
        Direction::HigherBetter,
    ),
    // Hot-cache (in-memory prompt-cache LRU) hit + eviction counters.
    (
        "prompt_cache_hot_cache_hits",
        "count",
        Direction::HigherBetter,
    ),
    (
        "prompt_cache_hot_cache_evictions",
        "count",
        Direction::LowerBetter,
    ),
    // SSD-tier hydrate hits — RAM misses served from the on-disk
    // `.kvb` tier (reader + index). Higher = more cross-process /
    // post-eviction reuse recovered from disk instead of re-prefilled.
    ("prompt_cache_ssd_hits", "count", Direction::HigherBetter),
    // Per-phase load-time spans.
    ("load_mmap_ms", "ms", Direction::LowerBetter),
    ("load_dequant_ms", "ms", Direction::LowerBetter),
    ("load_gpu_residency_ms", "ms", Direction::LowerBetter),
    ("load_first_kernel_ready_ms", "ms", Direction::LowerBetter),
    ("load_total_ms", "ms", Direction::LowerBetter),
    // C5 Slice A: FIFO admission-queue observability (rmlx-only).
    ("queue_wait_ms", "ms", Direction::LowerBetter),
    ("queue_depth", "count", Direction::LowerBetter),
    // F1b: per-request token counts from live HTTP handler (rmlx-only).
    // Suffixed `_live` to distinguish from the bench-config `prompt_tokens` /
    // `completion_tokens` columns on `observations` (those are bench metadata;
    // these are per-request telemetry metrics).
    ("prompt_tokens_live", "count", Direction::LowerBetter),
    ("completion_tokens_live", "count", Direction::LowerBetter),
    // F9: extended ITL percentile and spike counter (rmlx-only).
    // `itl_p99_ms` — 99th-percentile inter-token latency; emitted alongside
    // the existing itl_p50_ms/itl_p95_ms from `ItlStats`.
    // `itl_spikes` — count of intervals > 3×median per request; diagnostic for
    // GC pauses, Metal pipeline stalls, and other decode-path hiccups.
    ("itl_p99_ms", "ms", Direction::LowerBetter),
    ("itl_spikes", "count", Direction::LowerBetter),
    // Speculative-decoding metrics. Higher accept_rate = more draft tokens
    // survive verification per round. *_total counters monotonically grow per
    // request and are reset per run. `accepted_per_step` = total_accept /
    // n_rounds (mean draft tokens accepted per verifier step).
    ("accept_rate", "ratio", Direction::HigherBetter),
    ("draft_tokens_total", "count", Direction::HigherBetter),
    ("accept_tokens_total", "count", Direction::HigherBetter),
    ("draft_rounds_total", "count", Direction::HigherBetter),
    ("accepted_per_step", "ratio", Direction::HigherBetter),
    // SSD-tier observability (step2): byte/evict gauges + spill/hydrate timing.
    //
    // `ssd_bytes_used` is LowerBetter: unbounded cache growth is a budget risk.
    // A size shrink means eviction pressure; a size growth means the cache is
    // accumulating useful blocks. Either way, *unexpected* size changes are the
    // regression signal — LowerBetter keeps the gate from alerting on a known
    // cache fill (size > 0 is expected) while still flagging runaway growth.
    ("ssd_bytes_used", "bytes", Direction::LowerBetter),
    // More evictions = more cache thrash (LRU budget too tight or write-rate
    // too high). Lower is better.
    ("ssd_evict_total", "count", Direction::LowerBetter),
    // Spill / hydrate raw per-event latency (drain thread / request thread).
    // Each SQLite observation row carries one raw sample; real percentiles come
    // from the Prometheus histogram (rmlx_ssd_spill_us_bucket / rmlx_ssd_hydrate_us_bucket).
    // reason: a single-sample distribution has no meaningful p50 or p99 —
    // emitting the same dur_ms value as both is misleading; removed in H2 fix.
    ("ssd_spill_ms", "ms", Direction::LowerBetter),
    ("ssd_hydrate_ms", "ms", Direction::LowerBetter),
    // Spill / hydrate throughput. Higher = faster disk path.
    ("ssd_spill_mb_per_s", "mb/s", Direction::HigherBetter),
    ("ssd_hydrate_mb_per_s", "mb/s", Direction::HigherBetter),
    // -- offline perplexity scorer (`rmlx eval ppl`). Op family `ppl`;
    // one `ppl_<corpus>` metric per supported corpus + audit fields.
    ("ppl_wikitext2", "ppl", Direction::LowerBetter),
    ("ppl_mean_nll", "nat", Direction::LowerBetter),
    ("ppl_scored_tokens", "count", Direction::HigherBetter),
    ("ppl_windows", "count", Direction::HigherBetter),
    ("ppl_score_ms", "ms", Direction::LowerBetter),
    // phase-split TTFT/TPOT metrics. `prefill_duration_ms` reports the
    // wall-clock from generate-entry to first-OK-token (same value as the
    // existing `ttft_{warm,cold}_ms` row, but named as its own op so honest
    // stage attribution is possible without parsing `op` for "ttft" prefix).
    // `tpot_*_ms` mirrors `itl_*_ms` numerically in v1 (both definitions skip
    // the first interval = pure decode-only intervals); the value of the new
    // name is the *convention* + an open door for divergence (e.g. tpot
    // might one day exclude tool-calling stalls). `ttft_*_ms` / `itl_*_ms`
    // are kept for backward-compatibility.
    ("prefill_duration_ms", "ms", Direction::LowerBetter),
    ("tpot_p50_ms", "ms", Direction::LowerBetter),
    ("tpot_p95_ms", "ms", Direction::LowerBetter),
    ("tpot_p99_ms", "ms", Direction::LowerBetter),
];

/// Returns `(unit, direction)` for a known metric name.
pub fn lookup(name: &str) -> Result<(&'static str, Direction)> {
    METRICS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, u, d)| (*u, *d))
        .ok_or_else(|| Error::UnknownMetric(name.to_string()))
}

// ── Coverage ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed enum — four coverage states from spec §4; adding a state requires updating the coverage matrix and doctor logic"
)]
/// Backend coverage state for a given metric (see docs/METRICS_DB.md §4).
pub enum Coverage {
    /// Backend measures and records this metric.
    Yes,
    /// Backend can measure it but rMLX-side wiring is not done yet.
    Todo,
    /// Backend genuinely cannot measure this metric.
    No,
    /// Metric is exposed by the backend but not yet wired into rMLX recording.
    Maybe,
}

/// (backend, metric, coverage). Listed in spec §4 backend coverage matrix.
/// Used by `rmlx metrics doctor` to flag suspicious gaps.
pub const COVERAGE_MATRIX: &[(&str, &str, Coverage)] = &[
    // ── rmlx ──────────────────────────────────────────────────────────────────
    ("rmlx", "decode_tps_warm", Coverage::Yes),
    ("rmlx", "decode_tps_cold", Coverage::Yes),
    ("rmlx", "prefill_tps", Coverage::Yes),
    ("rmlx", "overall_tps", Coverage::Yes),
    ("rmlx", "ttft_warm_ms", Coverage::Yes),
    ("rmlx", "ttft_cold_ms", Coverage::Yes),
    ("rmlx", "itl_p50_ms", Coverage::Yes),
    ("rmlx", "itl_p95_ms", Coverage::Yes),
    ("rmlx", "step_ms_mean", Coverage::Yes),
    ("rmlx", "model_load_ms", Coverage::Yes),
    ("rmlx", "peak_rss_mb", Coverage::Yes),
    ("rmlx", "metal_peak_alloc_mb", Coverage::Yes),
    ("rmlx", "kv_cache_bytes", Coverage::Yes),
    ("rmlx", "tps_per_gb_ram", Coverage::Yes),
    ("rmlx", "task_pass_at_1", Coverage::No),
    // N19: prompt-cache counters — rmlx-only (other backends don't expose this).
    ("rmlx", "prompt_cache_hits", Coverage::Yes),
    ("rmlx", "prompt_cache_misses", Coverage::Yes),
    ("rmlx", "prompt_cache_bytes", Coverage::Yes),
    // C6: block-level counters — rmlx-only.
    ("rmlx", "prompt_cache_block_hits", Coverage::Yes),
    ("rmlx", "prompt_cache_block_misses", Coverage::Yes),
    ("rmlx", "prompt_cache_partial_hits", Coverage::Yes),
    ("rmlx", "prompt_cache_hot_cache_hits", Coverage::Yes),
    ("rmlx", "prompt_cache_hot_cache_evictions", Coverage::Yes),
    // SSD-tier hydrate hits — rmlx-only.
    ("rmlx", "prompt_cache_ssd_hits", Coverage::Yes),
    // Per-phase load-time spans — rmlx-only (arch::load_model instrumentation).
    ("rmlx", "load_mmap_ms", Coverage::Yes),
    ("rmlx", "load_dequant_ms", Coverage::Yes),
    ("rmlx", "load_gpu_residency_ms", Coverage::Yes),
    ("rmlx", "load_first_kernel_ready_ms", Coverage::Yes),
    ("rmlx", "load_total_ms", Coverage::Yes),
    // C5 Slice A: admission-queue metrics — rmlx-only (other backends have
    // no in-process admission queue to observe).
    ("rmlx", "queue_wait_ms", Coverage::Yes),
    ("rmlx", "queue_depth", Coverage::Yes),
    // F1b: per-request live token counts — rmlx-only (emitted from the HTTP
    // handler after each completed request; other backends do not flow through
    // this drainer path).
    ("rmlx", "prompt_tokens_live", Coverage::Yes),
    ("rmlx", "completion_tokens_live", Coverage::Yes),
    // F9: extended ITL stats — rmlx-only (emitted from engine decode path;
    // other backends do not have access to per-step Instant timestamps).
    ("rmlx", "itl_p99_ms", Coverage::Yes),
    ("rmlx", "itl_spikes", Coverage::Yes),
    // Speculative-decoding metrics — rmlx-only (other backends in this matrix
    // do not currently emit per-request spec stats; mlx_lm/paroquant/omlx run
    // non-spec, llama_cpp self-spec is not wired into the ingest pipeline).
    ("rmlx", "accept_rate", Coverage::Yes),
    ("rmlx", "draft_tokens_total", Coverage::Yes),
    ("rmlx", "accept_tokens_total", Coverage::Yes),
    ("rmlx", "draft_rounds_total", Coverage::Yes),
    ("rmlx", "accepted_per_step", Coverage::Yes),
    // SSD-tier observability — rmlx-only (other backends have no MLX-native
    // SSD KV-cache tier; coverage reflects the rmlx kv_cache/spill+hydrate
    // instrumentation added in step2).
    ("rmlx", "ssd_bytes_used", Coverage::Yes),
    ("rmlx", "ssd_evict_total", Coverage::Yes),
    ("rmlx", "ssd_spill_ms", Coverage::Yes),
    ("rmlx", "ssd_hydrate_ms", Coverage::Yes),
    ("rmlx", "ssd_spill_mb_per_s", Coverage::Yes),
    ("rmlx", "ssd_hydrate_mb_per_s", Coverage::Yes),
    // phase-split TTFT/TPOT — rmlx-only (emitted from the HTTP-handler
    // TTFT site and the engine ITL aggregation site; other backends do not
    // flow through these emit paths in v1).
    ("rmlx", "prefill_duration_ms", Coverage::Yes),
    ("rmlx", "tpot_p50_ms", Coverage::Yes),
    ("rmlx", "tpot_p95_ms", Coverage::Yes),
    ("rmlx", "tpot_p99_ms", Coverage::Yes),
    // ── mlx_lm ────────────────────────────────────────────────────────────────
    ("mlx_lm", "decode_tps_warm", Coverage::Yes),
    ("mlx_lm", "decode_tps_cold", Coverage::Yes),
    ("mlx_lm", "prefill_tps", Coverage::Yes),
    ("mlx_lm", "overall_tps", Coverage::Yes),
    ("mlx_lm", "ttft_warm_ms", Coverage::Yes),
    ("mlx_lm", "ttft_cold_ms", Coverage::Yes),
    ("mlx_lm", "itl_p50_ms", Coverage::Yes),
    ("mlx_lm", "itl_p95_ms", Coverage::Yes),
    ("mlx_lm", "step_ms_mean", Coverage::Yes),
    ("mlx_lm", "model_load_ms", Coverage::Yes),
    ("mlx_lm", "peak_rss_mb", Coverage::Yes),
    ("mlx_lm", "metal_peak_alloc_mb", Coverage::Yes),
    ("mlx_lm", "kv_cache_bytes", Coverage::No),
    ("mlx_lm", "tps_per_gb_ram", Coverage::Yes),
    ("mlx_lm", "task_pass_at_1", Coverage::No),
    // ── paroquant ─────────────────────────────────────────────────────────────
    ("paroquant", "decode_tps_warm", Coverage::Yes),
    ("paroquant", "decode_tps_cold", Coverage::Yes),
    ("paroquant", "prefill_tps", Coverage::Yes),
    ("paroquant", "overall_tps", Coverage::Yes),
    ("paroquant", "ttft_warm_ms", Coverage::Yes),
    ("paroquant", "ttft_cold_ms", Coverage::Yes),
    ("paroquant", "itl_p50_ms", Coverage::Yes),
    ("paroquant", "itl_p95_ms", Coverage::Yes),
    ("paroquant", "step_ms_mean", Coverage::Yes),
    ("paroquant", "model_load_ms", Coverage::Yes),
    ("paroquant", "peak_rss_mb", Coverage::Yes),
    ("paroquant", "metal_peak_alloc_mb", Coverage::Yes),
    ("paroquant", "kv_cache_bytes", Coverage::No),
    ("paroquant", "tps_per_gb_ram", Coverage::Yes),
    ("paroquant", "task_pass_at_1", Coverage::No),
    // ── omlx ──────────────────────────────────────────────────────────────────
    ("omlx", "decode_tps_warm", Coverage::Yes),
    ("omlx", "decode_tps_cold", Coverage::Yes),
    ("omlx", "prefill_tps", Coverage::Yes),
    ("omlx", "overall_tps", Coverage::Yes),
    ("omlx", "ttft_warm_ms", Coverage::Yes),
    ("omlx", "ttft_cold_ms", Coverage::Yes),
    ("omlx", "itl_p50_ms", Coverage::Yes),
    ("omlx", "itl_p95_ms", Coverage::Yes),
    ("omlx", "step_ms_mean", Coverage::Yes),
    ("omlx", "model_load_ms", Coverage::Yes),
    ("omlx", "peak_rss_mb", Coverage::Yes),
    ("omlx", "metal_peak_alloc_mb", Coverage::Yes),
    ("omlx", "kv_cache_bytes", Coverage::Maybe),
    ("omlx", "tps_per_gb_ram", Coverage::Yes),
    ("omlx", "task_pass_at_1", Coverage::No),
    // ── llama_cpp ─────────────────────────────────────────────────────────────
    // llama-bench -o json exposes pp (prefill TPS) and tg (decode TPS) natively.
    // Metal / CPU load time measurable via wall clock around model load.
    // metal_peak_alloc_mb: no MLX Metal allocator; llama.cpp uses its own pool.
    // kv_cache_bytes: not directly exposed by llama-bench JSON output.
    // task_pass_at_1: quality probe, not a bench metric.
    ("llama_cpp", "decode_tps_warm", Coverage::Yes),
    ("llama_cpp", "decode_tps_cold", Coverage::Yes),
    ("llama_cpp", "prefill_tps", Coverage::Yes),
    ("llama_cpp", "overall_tps", Coverage::No),
    ("llama_cpp", "ttft_warm_ms", Coverage::No),
    ("llama_cpp", "ttft_cold_ms", Coverage::No),
    ("llama_cpp", "itl_p50_ms", Coverage::No),
    ("llama_cpp", "itl_p95_ms", Coverage::No),
    ("llama_cpp", "step_ms_mean", Coverage::Yes),
    ("llama_cpp", "model_load_ms", Coverage::Yes),
    ("llama_cpp", "peak_rss_mb", Coverage::Yes),
    ("llama_cpp", "metal_peak_alloc_mb", Coverage::No),
    ("llama_cpp", "kv_cache_bytes", Coverage::No),
    ("llama_cpp", "tps_per_gb_ram", Coverage::Yes),
    ("llama_cpp", "task_pass_at_1", Coverage::No),
    // ── ollama ────────────────────────────────────────────────────────────────
    ("ollama", "decode_tps_warm", Coverage::Yes),
    ("ollama", "decode_tps_cold", Coverage::Yes),
    ("ollama", "prefill_tps", Coverage::Yes),
    ("ollama", "overall_tps", Coverage::Yes),
    ("ollama", "ttft_warm_ms", Coverage::Yes),
    ("ollama", "ttft_cold_ms", Coverage::Yes),
    ("ollama", "itl_p50_ms", Coverage::Yes),
    ("ollama", "itl_p95_ms", Coverage::Yes),
    ("ollama", "step_ms_mean", Coverage::Yes),
    ("ollama", "model_load_ms", Coverage::Yes),
    ("ollama", "peak_rss_mb", Coverage::Yes),
    ("ollama", "metal_peak_alloc_mb", Coverage::No),
    ("ollama", "kv_cache_bytes", Coverage::No),
    ("ollama", "tps_per_gb_ram", Coverage::Yes),
    ("ollama", "task_pass_at_1", Coverage::No),
];

/// Returns the coverage for a (backend, metric) pair.
/// Unknown pairs return `Coverage::No` — if it's not listed, it's unsupported.
pub fn coverage(backend: &str, metric: &str) -> Coverage {
    COVERAGE_MATRIX
        .iter()
        .find(|(b, m, _)| *b == backend && *m == metric)
        .map_or(Coverage::No, |(_, _, c)| *c)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
