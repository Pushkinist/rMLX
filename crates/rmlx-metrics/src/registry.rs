//! Metric registry per `docs/METRICS_DB.md` §4.
//!
//! Maps metric names to `(unit, direction, bounds)` triples. The registry is
//! the authoritative source for what constitutes a valid metric name, how to
//! interpret its direction (`Higher` = better, `Lower` = better), and which
//! values are physically possible measurements of it. Unregistered metric
//! names, and values outside the bounds, are rejected at ingest time.
//!
//! # Public API
//!
//! - [`lookup`] — resolve a metric name to its unit string and [`Direction`].
//! - [`bounds`] — resolve a metric name to its plausible-value [`Bounds`].
//! - [`Direction`] — `Higher` or `Lower` (better) enum.
//! - [`Bounds`] — the plausible-value window for one metric.
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

// ── Bounds ────────────────────────────────────────────────────────────────────

/// The window of values that can be a *measurement* of a metric.
///
/// Every metric in the registry counts, times or rates a physical quantity, so
/// a negative value is never a measurement and the floor is always `0.0`. What
/// differs per metric is whether that floor is itself a measurement:
///
/// * A **rate** cannot be `0.0`. `tokens / seconds` is zero only when no token
///   was produced, i.e. when there was nothing to measure — the zero is a
///   missing field wearing a number's clothes.
/// * A **counter, duration or gauge** can be `0.0` honestly: zero cache hits,
///   a sub-millisecond span rounded down, a run that allocated no Metal.
///
/// The ceiling is the same idea from the other end: a value orders of magnitude
/// past anything the hardware can do is an arithmetic accident, not a record.
/// Ceilings are deliberately loose (several × the best value ever recorded) so
/// they reject fabrications, not fast machines.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "closed value-window struct — a ceiling plus whether zero counts is the whole contract; a signed metric would need a `min` field and a registry-wide review"
)]
pub struct Bounds {
    /// Largest plausible value, inclusive.
    pub max: f64,
    /// Whether an exact `0.0` is a measurement rather than a missing field.
    pub zero_is_measurement: bool,
}

impl Bounds {
    /// A quantity whose zero means "nothing was measured" — every rate, plus
    /// gauges no live run can read as zero (a running process has RSS).
    pub const fn positive(max: f64) -> Self {
        Self {
            max,
            zero_is_measurement: false,
        }
    }

    /// A quantity that can legitimately read `0.0` — counters, durations
    /// (millisecond resolution rounds sub-ms spans to zero), and gauges that
    /// a run can genuinely leave at zero.
    pub const fn non_negative(max: f64) -> Self {
        Self {
            max,
            zero_is_measurement: true,
        }
    }

    /// Whether `value` is inside the window. Rejects NaN and infinities.
    pub fn contains(self, value: f64) -> bool {
        if !value.is_finite() || value > self.max {
            return false;
        }
        if self.zero_is_measurement {
            value >= 0.0
        } else {
            value > 0.0
        }
    }

    /// Renders the window as a SQLite boolean expression over `column`.
    ///
    /// Used to build the `bests` view from this registry so the view and the
    /// ingest gate cannot disagree about what a measurement is.
    pub fn sql(self, column: &str) -> String {
        let floor = if self.zero_is_measurement { ">=" } else { ">" };
        format!("{column} {floor} 0.0 AND {column} <= {:?}", self.max)
    }

    /// One-line human description, e.g. `"(0, 100000.0]"`.
    pub fn describe(self) -> String {
        let open = if self.zero_is_measurement { '[' } else { '(' };
        format!("{open}0, {:?}]", self.max)
    }
}

/// One hour, in milliseconds — the ceiling for every duration metric. A single
/// span longer than this is a hung run, not a measurement.
const MS_CEILING: f64 = 3_600_000.0;
/// Ceiling for monotonic counters (cache hits, token counts, spike counts).
const COUNT_CEILING: f64 = 1e12;
/// Ceiling for byte gauges — 10 TB, well past unified memory on any Mac.
const BYTES_CEILING: f64 = 1e13;
/// Ceiling for megabyte gauges — 1 PB expressed in MB.
const MB_CEILING: f64 = 1e9;

// ── METRICS const ─────────────────────────────────────────────────────────────

/// What a speculative metric is: a raw total over the request, or a figure
/// derived from those totals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "a speculative metric is a counter or something computed from counters; a third kind would be a different table"
)]
pub enum SpecRole {
    /// Cumulative over the request. Recorded for audit; not a table column,
    /// because a total says nothing without the rounds it is spread over.
    Counter,
    /// Computed from the counters. What the table shows.
    Derived,
}

/// The metrics a speculative round loop reports, in the order they are derived.
///
/// One declaration of the set. `rmlx_metrics::export`'s speculative table
/// renders exactly the [`SpecRole::Derived`] ones (plus the throughput they
/// explain) and a test pins that, so a metric added here reaches the table
/// rather than being invisible until somebody notices the column is missing.
///
/// `decode_tps_warm` is deliberately not here: every backend emits it and it is
/// not a round-loop figure.
pub const SPEC_METRICS: &[(&str, SpecRole)] = &[
    ("accept_rate", SpecRole::Derived),
    ("draft_tokens_total", SpecRole::Counter),
    ("accept_tokens_total", SpecRole::Counter),
    ("draft_rounds_total", SpecRole::Counter),
    ("accepted_per_step", SpecRole::Derived),
    ("tokens_per_round", SpecRole::Derived),
    ("draft_ms_per_round", SpecRole::Derived),
    ("verify_ms_per_round", SpecRole::Derived),
    ("loop_ms_per_round", SpecRole::Derived),
];

/// Canonical metric name → (unit, direction, plausible bounds).
/// Add new metrics here AND in §4.
pub const METRICS: &[(&str, &str, Direction, Bounds)] = &[
    (
        "decode_tps_warm",
        "tps",
        Direction::HigherBetter,
        Bounds::positive(1e4),
    ),
    (
        "decode_tps_cold",
        "tps",
        Direction::HigherBetter,
        Bounds::positive(1e4),
    ),
    (
        "prefill_tps",
        "tps",
        Direction::HigherBetter,
        Bounds::positive(1e5),
    ),
    // Same ceiling as `decode_tps_warm`, and not a looser one: `overall_tps`
    // divides the same token count by a wall clock that also contains prefill,
    // so it is bounded above by the decode rate by construction.
    (
        "overall_tps",
        "tps",
        Direction::HigherBetter,
        Bounds::positive(1e4),
    ),
    (
        "ttft_warm_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "ttft_cold_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "itl_p50_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "itl_p95_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "step_ms_mean",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "model_load_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "peak_rss_mb",
        "mb",
        Direction::LowerBetter,
        Bounds::positive(MB_CEILING),
    ),
    (
        "peak_phys_footprint_mb",
        "mb",
        Direction::LowerBetter,
        Bounds::positive(MB_CEILING),
    ),
    (
        "metal_peak_alloc_mb",
        "mb",
        Direction::LowerBetter,
        Bounds::non_negative(MB_CEILING),
    ),
    (
        "kv_cache_bytes",
        "bytes",
        Direction::LowerBetter,
        Bounds::non_negative(BYTES_CEILING),
    ),
    (
        "tps_per_gb_ram",
        "ratio",
        Direction::HigherBetter,
        Bounds::positive(1e5),
    ),
    (
        "task_pass_at_1",
        "ratio",
        Direction::HigherBetter,
        Bounds::non_negative(1.0),
    ),
    // N19: prompt-cache hit/miss/bytes counters.
    (
        "prompt_cache_hits",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "prompt_cache_misses",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "prompt_cache_bytes",
        "bytes",
        Direction::LowerBetter,
        Bounds::non_negative(BYTES_CEILING),
    ),
    // C6: block-level prompt-cache counters (monotonic).
    (
        "prompt_cache_block_hits",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "prompt_cache_block_misses",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "prompt_cache_partial_hits",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // Hot-cache (in-memory prompt-cache LRU) hit + eviction counters.
    (
        "prompt_cache_hot_cache_hits",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "prompt_cache_hot_cache_evictions",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // SSD-tier hydrate hits — RAM misses served from the on-disk
    // `.kvb` tier (reader + index). Higher = more cross-process /
    // post-eviction reuse recovered from disk instead of re-prefilled.
    (
        "prompt_cache_ssd_hits",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // Per-phase load-time spans.
    (
        "load_mmap_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "load_dequant_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "load_gpu_residency_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "load_first_kernel_ready_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "load_total_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    // C5 Slice A: FIFO admission-queue observability (rmlx-only).
    (
        "queue_wait_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "queue_depth",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // F1b: per-request token counts from live HTTP handler (rmlx-only).
    // Suffixed `_live` to distinguish from the bench-config `prompt_tokens` /
    // `completion_tokens` columns on `observations` (those are bench metadata;
    // these are per-request telemetry metrics).
    (
        "prompt_tokens_live",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "completion_tokens_live",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // F9: extended ITL percentile and spike counter (rmlx-only).
    // `itl_p99_ms` — 99th-percentile inter-token latency; emitted alongside
    // the existing itl_p50_ms/itl_p95_ms from `ItlStats`.
    // `itl_spikes` — count of intervals > 3×median per request; diagnostic for
    // GC pauses, Metal pipeline stalls, and other decode-path hiccups.
    (
        "itl_p99_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "itl_spikes",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // Speculative-decoding metrics. Higher accept_rate = more draft tokens
    // survive verification per round. *_total counters monotonically grow per
    // request and are reset per run. `accepted_per_step` = total_accept /
    // n_rounds (mean draft tokens accepted per verifier step).
    //
    // `tokens_per_round` is the one that lets a speculative result be read
    // independently of the kernels: accepted drafts plus the verifier's own
    // token. It is `1 + accept_rate * (block - 1)` only while every round
    // drafts the configured block, which an adaptive drafter does not, so it is
    // recorded rather than derived at read time.
    //
    // The three `*_ms_per_round` figures partition one round's wall clock:
    // drafting, verifying, and everything else the loop does (rollback,
    // snapshot and restore, acceptance walks, sampling). The third is the
    // round-loop overhead itself.
    (
        "accept_rate",
        "ratio",
        Direction::HigherBetter,
        Bounds::non_negative(1.0),
    ),
    (
        "draft_tokens_total",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "accept_tokens_total",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "draft_rounds_total",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "accepted_per_step",
        "ratio",
        Direction::HigherBetter,
        Bounds::non_negative(1e3),
    ),
    (
        "tokens_per_round",
        "ratio",
        Direction::HigherBetter,
        Bounds::non_negative(1e3),
    ),
    (
        "draft_ms_per_round",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "verify_ms_per_round",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "loop_ms_per_round",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    // SSD-tier observability (step2): byte/evict gauges + spill/hydrate timing.
    //
    // `ssd_bytes_used` is LowerBetter: unbounded cache growth is a budget risk.
    // A size shrink means eviction pressure; a size growth means the cache is
    // accumulating useful blocks. Either way, *unexpected* size changes are the
    // regression signal — LowerBetter keeps the gate from alerting on a known
    // cache fill (size > 0 is expected) while still flagging runaway growth.
    (
        "ssd_bytes_used",
        "bytes",
        Direction::LowerBetter,
        Bounds::non_negative(BYTES_CEILING),
    ),
    // More evictions = more cache thrash (LRU budget too tight or write-rate
    // too high). Lower is better.
    (
        "ssd_evict_total",
        "count",
        Direction::LowerBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    // Spill / hydrate raw per-event latency (drain thread / request thread).
    // Each SQLite observation row carries one raw sample; real percentiles come
    // from the Prometheus histogram (rmlx_ssd_spill_us_bucket / rmlx_ssd_hydrate_us_bucket).
    // reason: a single-sample distribution has no meaningful p50 or p99 —
    // emitting the same dur_ms value as both is misleading; removed in H2 fix.
    (
        "ssd_spill_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "ssd_hydrate_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    // Spill / hydrate throughput. Higher = faster disk path.
    (
        "ssd_spill_mb_per_s",
        "mb/s",
        Direction::HigherBetter,
        Bounds::positive(1e6),
    ),
    (
        "ssd_hydrate_mb_per_s",
        "mb/s",
        Direction::HigherBetter,
        Bounds::positive(1e6),
    ),
    // -- offline perplexity scorer (`rmlx eval ppl`). Op family `ppl`;
    // one `ppl_<corpus>` metric per supported corpus + audit fields.
    //
    // The scorer has two modes and they do not measure the same quantity: the
    // default forwards each window once with no KV cache, and `--kv-quant`
    // teacher-forces the window through a real per-layer cache, one forward
    // per scored token. `_cached` is therefore a metric of its own rather than
    // a `decode_config` term — a term would also fence these rows off from
    // every `mlx_lm` row, which can never carry one this engine invented.
    (
        "ppl_wikitext2",
        "ppl",
        Direction::LowerBetter,
        Bounds::positive(1e6),
    ),
    (
        "ppl_wikitext2_cached",
        "ppl",
        Direction::LowerBetter,
        Bounds::positive(1e6),
    ),
    (
        "ppl_mean_nll",
        "nat",
        Direction::LowerBetter,
        Bounds::non_negative(1e3),
    ),
    (
        "ppl_scored_tokens",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "ppl_windows",
        "count",
        Direction::HigherBetter,
        Bounds::non_negative(COUNT_CEILING),
    ),
    (
        "ppl_score_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    // phase-split TTFT/TPOT metrics. `prefill_duration_ms` reports the
    // wall-clock from generate-entry to first-OK-token (same value as the
    // existing `ttft_{warm,cold}_ms` row, but named as its own op so honest
    // stage attribution is possible without parsing `op` for "ttft" prefix).
    // `tpot_*_ms` mirrors `itl_*_ms` numerically in v1 (both definitions skip
    // the first interval = pure decode-only intervals); the value of the new
    // name is the *convention* + an open door for divergence (e.g. tpot
    // might one day exclude tool-calling stalls). `ttft_*_ms` / `itl_*_ms`
    // are kept for backward-compatibility.
    (
        "prefill_duration_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "tpot_p50_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "tpot_p95_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
    (
        "tpot_p99_ms",
        "ms",
        Direction::LowerBetter,
        Bounds::non_negative(MS_CEILING),
    ),
];

/// Returns `(unit, direction)` for a known metric name.
pub fn lookup(name: &str) -> Result<(&'static str, Direction)> {
    METRICS
        .iter()
        .find(|(n, _, _, _)| *n == name)
        .map(|(_, u, d, _)| (*u, *d))
        .ok_or_else(|| Error::UnknownMetric(name.to_string()))
}

/// Returns the plausible-value [`Bounds`] for a known metric name.
pub fn bounds(name: &str) -> Result<Bounds> {
    METRICS
        .iter()
        .find(|(n, _, _, _)| *n == name)
        .map(|(_, _, _, b)| *b)
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

/// The per-backend metric spec: the metrics every wired backend must declare a
/// [`Coverage`] for, whether `Yes` or `No`.
///
/// `coverage()` falls back to `No` for a pair it cannot find, so an omitted row
/// and a measured "this backend cannot emit that" are the same answer. Naming
/// the set here is what lets a test tell them apart.
pub const BACKEND_METRIC_SPEC: &[&str] = &[
    "decode_tps_warm",
    "decode_tps_cold",
    "prefill_tps",
    "overall_tps",
    "ttft_warm_ms",
    "ttft_cold_ms",
    "itl_p50_ms",
    "itl_p95_ms",
    "step_ms_mean",
    "model_load_ms",
    "peak_rss_mb",
    "metal_peak_alloc_mb",
    "kv_cache_bytes",
    "tps_per_gb_ram",
    "task_pass_at_1",
];

/// Backends in [`crate::identity::BACKEND_WHITELIST`] that deliberately have no
/// rows in [`COVERAGE_MATRIX`] yet.
///
/// This list exists so that adding a backend is a *visible* act. Without it,
/// a new whitelist entry with no matrix rows makes `coverage()` answer `No` for
/// every metric — indistinguishable from "measured and genuinely unsupported" —
/// and nothing fails. Anything added here must be a backend nothing has
/// recorded yet; the moment it produces a row, wire its metrics instead.
pub const BACKENDS_WITHOUT_COVERAGE: &[&str] = &[
    // Declared "(future)" in docs/METRICS_DB.md §5.4; no runner, no rows.
    "vllm",
];

/// (backend, metric, coverage). Listed in spec §4 backend coverage matrix.
/// Used by `rmlx metrics doctor` to flag suspicious gaps.
///
/// Every backend in [`crate::identity::BACKEND_WHITELIST`] must appear here for
/// each metric in [`BACKEND_METRIC_SPEC`], unless it is declared in
/// [`BACKENDS_WITHOUT_COVERAGE`].
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
    ("rmlx", "tokens_per_round", Coverage::Yes),
    ("rmlx", "draft_ms_per_round", Coverage::Yes),
    ("rmlx", "verify_ms_per_round", Coverage::Yes),
    ("rmlx", "loop_ms_per_round", Coverage::Yes),
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
    // ── mlx_lm_tq (the mlx-lm TurboQuant fork) ────────────────────────────────
    // Whitelisted and recording since the cross-backend campaigns, but it had no
    // COVERAGE_MATRIX rows at all until the sweep below was driven off
    // BACKEND_WHITELIST — so `coverage()` answered No for every metric on the
    // second-most-recorded backend in the store. Same CBB runner and same
    // OpenAI-compatible surface as `mlx_lm`, hence the same coverage, except
    // that its whole reason to exist is a quantised KV cache it does not report
    // the size of.
    ("mlx_lm_tq", "decode_tps_warm", Coverage::Yes),
    ("mlx_lm_tq", "decode_tps_cold", Coverage::Yes),
    ("mlx_lm_tq", "prefill_tps", Coverage::Yes),
    ("mlx_lm_tq", "overall_tps", Coverage::Yes),
    ("mlx_lm_tq", "ttft_warm_ms", Coverage::Yes),
    ("mlx_lm_tq", "ttft_cold_ms", Coverage::Yes),
    ("mlx_lm_tq", "itl_p50_ms", Coverage::Yes),
    ("mlx_lm_tq", "itl_p95_ms", Coverage::Yes),
    ("mlx_lm_tq", "step_ms_mean", Coverage::Yes),
    ("mlx_lm_tq", "model_load_ms", Coverage::Yes),
    ("mlx_lm_tq", "peak_rss_mb", Coverage::Yes),
    ("mlx_lm_tq", "metal_peak_alloc_mb", Coverage::Yes),
    ("mlx_lm_tq", "kv_cache_bytes", Coverage::No),
    ("mlx_lm_tq", "tps_per_gb_ram", Coverage::Yes),
    ("mlx_lm_tq", "task_pass_at_1", Coverage::No),
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
    // Two producers feed this backend and they cover different metrics:
    // `llama_bench_ingest.py` reads `llama-bench -o json` (pp = prefill TPS,
    // tg = decode TPS, nothing else), and `llama_ab_ingest.py` reads a
    // `bench_llama_ab.sh` result, which carries the server's own `timings`
    // plus the KV-buffer total parsed from the server log and sampled peak RSS.
    // A cell is Yes when EITHER producer can supply it.
    // metal_peak_alloc_mb: no MLX Metal allocator; llama.cpp uses its own pool.
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
    // Yes since the A/B harness landed: `llama_kv_cache: ... KV buffer size`
    // is on the server's own startup log and is summed per slot. It is not in
    // `llama-bench` JSON, which is why this cell used to read No.
    ("llama_cpp", "kv_cache_bytes", Coverage::Yes),
    ("llama_cpp", "tps_per_gb_ram", Coverage::Yes),
    ("llama_cpp", "task_pass_at_1", Coverage::No),
    // ── llama_cpp_tq (the llama-cpp-turboquant fork) ──────────────────────────
    // Same server, same log surface, same producer — it differs from upstream
    // only in the KV codecs it can load, so its coverage is identical.
    ("llama_cpp_tq", "decode_tps_warm", Coverage::Yes),
    ("llama_cpp_tq", "decode_tps_cold", Coverage::Yes),
    ("llama_cpp_tq", "prefill_tps", Coverage::Yes),
    ("llama_cpp_tq", "overall_tps", Coverage::No),
    ("llama_cpp_tq", "ttft_warm_ms", Coverage::No),
    ("llama_cpp_tq", "ttft_cold_ms", Coverage::No),
    ("llama_cpp_tq", "itl_p50_ms", Coverage::No),
    ("llama_cpp_tq", "itl_p95_ms", Coverage::No),
    ("llama_cpp_tq", "step_ms_mean", Coverage::Yes),
    ("llama_cpp_tq", "model_load_ms", Coverage::Yes),
    ("llama_cpp_tq", "peak_rss_mb", Coverage::Yes),
    ("llama_cpp_tq", "metal_peak_alloc_mb", Coverage::No),
    ("llama_cpp_tq", "kv_cache_bytes", Coverage::Yes),
    ("llama_cpp_tq", "tps_per_gb_ram", Coverage::Yes),
    ("llama_cpp_tq", "task_pass_at_1", Coverage::No),
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
