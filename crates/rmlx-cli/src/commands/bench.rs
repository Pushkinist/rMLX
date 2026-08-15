//! `rmlx bench` command implementation.
//!
//! A repeated-run decode instrument. One invocation serves the same
//! (model, KV codec, context, generation length) cell `--warmup` + `--runs`
//! times in-process and reports TTFT, inter-token latency (ITL p50/p99),
//! steady-state decode TPS and filled-prefix KV bytes as a median plus the
//! observed run-to-run range.
//!
//! Two properties matter more than the feature list, because the numbers this
//! prints are used to accept or reject other work:
//!
//! 1. **It refuses to print a number it cannot stand behind.** Every quantity
//!    it reports is checked against the condition that makes it meaningful, and
//!    a failed check aborts the run with the reason. In particular a KV-byte
//!    figure that the just-finished generation did not itself report, and a
//!    generation whose prefill was served from the prompt cache (so its TTFT is
//!    not a time-to-first-token), are hard errors — not a fast-looking number.
//! 2. **It never reports a bare central value.** Every metric carries its
//!    observed min/max across the measured runs, and `--runs` is rejected below
//!    2 so there is always a spread to report. A metric that never settled
//!    across those runs has no central value at all, and is refused rather than
//!    summarised — whether it settled is decided by two independent gates, one
//!    per shape a cell fails to settle in: a *trend* (see [`detect_drift`]) and
//!    a *range* too wide to call noise around a centre (see
//!    [`SETTLED_MAX_RANGE_PCT`]). Which metrics are gated is enumerated on
//!    [`Metric::gate`], not left to which call sites happen to check.
//!
//! `bench` is read-only with respect to the metrics database: it prints, it
//! does not record. Use `rmlx baseline --record` for the append-only store.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rmlx_mlx::Device;
use rmlx_models::{arch, classify_kv_bytes, CacheStats, KvBytesVerdict};
use tracing::{info, warn};

use super::baseline::tokenize_prompt_text;

/// Minimum number of measured runs.
///
/// A single measurement has no observable spread, and a decode-TPS figure with
/// no spread has repeatedly supported conclusions that a second run overturned.
/// The instrument therefore has no single-run mode.
pub(crate) const MIN_RUNS: u32 = 2;

/// Prompt-cache slots used for every bench generation.
///
/// One — an ordinary single-slot cache, the same shape `rmlx baseline` serves
/// under. `bench` runs N generations of the *same* prompt in one process, so
/// run 2 onwards would be served from the prompt cache, skip prefill entirely,
/// and report a TTFT of a few milliseconds that looks like a very fast prefill
/// instead of a measurement that was never taken.
///
/// What prevents that is [`arch::Architecture::clear_prompt_cache`], called
/// before every generation: the RAM slots are emptied, so the next request
/// misses. Asking for *zero* slots would also miss every time, but it would
/// measure a cache no operator runs — the point is to time the configuration
/// that gets served, with its snapshots dropped between runs.
///
/// A RAM miss is not the same as a prefill: the clear does not detach an SSD KV
/// source, which can serve the miss from a `.kvb` instead. So
/// `assert_prefill_measured` checks the outcome of every run rather than
/// trusting either the constant or the clear.
const BENCH_PROMPT_CACHE_SLOTS: usize = 1;

/// Largest run-to-run trend a metric may show and still be summarised, as a
/// percentage of its median. See [`detect_drift`].
///
/// Ten percent, calibrated against what this instrument is used for: it accepts
/// or rejects changes at the repo's ±1% decode-TPS band and its >5%
/// regression-stop line. A cell whose own metric moves more than twice that
/// line across its own runs cannot support either comparison, whichever way it
/// moves — an improving ramp means the cell had not converged just as much as a
/// degrading one does.
///
/// This is a floor, not the whole test: clearing it says the fitted change is
/// *large*, not that it is a trend. See [`TREND_MIN_RESID_MULTIPLE`].
const TREND_MAX_DRIFT_PCT: f64 = 10.0;

/// How far the fitted change must exceed the cell's own residual scatter before
/// it is called a trend.
///
/// The percentage floor alone has no noise anchoring, and at the default
/// `--runs 3` the fit is thin enough for that to matter: with three runs the
/// least-squares change reduces exactly to `last − first`, so the middle run
/// contributes nothing but its share of the median and a single zig-zag reads
/// as a slope. `[100, 118, 88]` fits to −12% and would be refused as a decline
/// it plainly is not.
///
/// Requiring the fitted change to also dominate the scatter *about* the fitted
/// line separates the two: on a real ramp the residuals are small next to the
/// change (a measured 17.8 s → 25.5 s TTFT ramp fits 6831 ms against a 2×RMS of
/// 1479), while on a zig-zag they are the same size as it (2×RMS 22.6 against a
/// fitted 12). Two is the multiple because at three runs a zig-zag has the fixed
/// residual pattern `(−k, +2k, −k)`, whose RMS is `k√2` — so its scatter is
/// always the same order as the step that produced it, and a multiple of two
/// puts the `[100, 118, 88]` case (2×RMS 22.6 vs fitted 12) on the refuse side
/// with margin, while leaving real ramps — whose residuals are a small fraction
/// of the change — several times clear of it.
///
/// Both conditions must hold. Refusing on scatter alone would abort settled
/// cells (a tight cell has tiny residuals *and* a tiny change), and refusing on
/// percentage alone is the noise-driven abort whose documented remedy — raise
/// `--warmup` and re-measure — is measure-until-it-passes.
const TREND_MIN_RESID_MULTIPLE: f64 = 2.0;

/// Largest observed range a gated metric may show and still be summarised, as a
/// percentage of its median. See [`Spread::range_pct`].
///
/// The trend gate only refuses movement that goes *one way*. A cell can fail to
/// settle without trending — a single-run spike has slope zero at any
/// magnitude, and a late-onset step fits to a smaller and smaller slope as
/// `--runs` grows (a last-run jump of `d` fits to `1.33d` at three runs but only
/// `0.29d` at twenty), so the operator's natural response to a noisy abort would
/// otherwise make the guard weaker. The range gate closes both: it is
/// order-blind on purpose, and refuses "did not settle" whichever shape it takes.
///
/// Fifteen percent, from this instrument's own measured settled cells — ten
/// cells across gemma-4-e2b and Ternary-Bonsai-8B at 4k, 32k and 64k, all
/// pinned in `a_settled_cell_clears_the_range_gate`. Most gated metrics of a
/// settled cell range under 2% (tightest: 0.17%), but two settled cells on that
/// same host produced a 7.56% TTFT range and a 7.09% decode-TPS range, and the
/// widest *accepted* wobble on record is the 8.35% TTFT range pinned in
/// `the_threshold_separates_warmup_wobble_from_a_ramp`. So the settled
/// population is not 2%-wide, it has a ~8% tail, and a ceiling near 10% would
/// leave that tail about 1.2× of margin — it would abort settled cells on a bad
/// day. Fifteen sits at 1.8× the worst accepted case while still refusing a
/// spike or a late step, which run to tens of percent.
///
/// Only [`Gate::Settled`] metrics are checked. `itl_p99_ms` in particular
/// ranged 58.6% on a cell whose every other metric ranged under 2% — an
/// extreme-order statistic has no settled range to speak of, which is why it is
/// reported rather than gated.
const SETTLED_MAX_RANGE_PCT: f64 = 15.0;

// ---------------------------------------------------------------------------
// Spread — a central value that carries its observed range
// ---------------------------------------------------------------------------

/// Median of a sample together with the range actually observed.
///
/// There is no constructor that drops `min`/`max`: a caller cannot obtain the
/// median of a bench metric without also obtaining its spread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spread {
    /// Median over the measured runs.
    pub median: f64,
    /// Smallest value observed.
    pub min: f64,
    /// Largest value observed.
    pub max: f64,
    /// Number of measured runs behind the figures.
    pub n: usize,
}

impl Spread {
    /// Summarise a sample. `None` for an empty sample — there is no median of
    /// nothing, and returning a zero would be indistinguishable from a real
    /// measurement of zero.
    ///
    /// Borrows: every caller also needs the values in collection order for the
    /// settle gates, so a by-value signature would have them clone first and
    /// allocate exactly as often.
    pub(crate) fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        // Bounds: `sorted` is non-empty (checked above), so `first`/`last` are
        // `Some` and the median indices are in range.
        let median = if n % 2 == 1 {
            *sorted.get(n / 2)?
        } else {
            f64::midpoint(*sorted.get(n / 2 - 1)?, *sorted.get(n / 2)?)
        };
        Some(Self {
            median,
            min: *sorted.first()?,
            max: *sorted.last()?,
            n,
        })
    }

    /// Observed range as a percentage of the median — the one number that says
    /// how much to trust the median. `0.0` when the median is zero (no scale to
    /// express the range against).
    pub(crate) fn range_pct(&self) -> f64 {
        if self.median == 0.0 {
            0.0
        } else {
            (self.max - self.min) / self.median.abs() * 100.0
        }
    }
}

/// Nearest-rank percentile of an already-sorted sample, `q` in `0.0..=1.0`.
///
/// `None` for an empty sample, for the same reason `Spread::of` returns `None`:
/// a percentile of nothing is not zero.
pub(crate) fn percentile_sorted(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&q) {
        return None;
    }
    let n = sorted.len();
    // Nearest-rank: rank = ceil(q * n), clamped to 1..=n, then 0-based.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "q is in 0..=1 and n is a slice length, so q*n is a small non-negative float"
    )]
    let rank = (q * n as f64).ceil() as usize;
    sorted.get(rank.clamp(1, n) - 1).copied()
}

// ---------------------------------------------------------------------------
// Drift — a trend is not a spread
// ---------------------------------------------------------------------------

/// Detect a run-to-run trend in `values`, **in collection order**.
///
/// Returns the fitted change from the first measured run to the last as a
/// signed percentage of the median — positive means the metric rose.
///
/// [`Spread`] is order-blind by construction: it sorts. A metric that climbs
/// monotonically from run to run therefore reports as a wide *spread*, and a
/// wide spread is read as noise around a central value. It is not — a drifting
/// cell has no central value, and its median is a point on a ramp that depends
/// on where the operator stopped measuring.
///
/// The test is the least-squares slope over the run index, scaled to the whole
/// run sequence and expressed against the median. Ordinary regression rather
/// than a monotonicity test on purpose: a real ramp with one noisy step is
/// still a ramp, and would escape a strict-monotonic check once `--runs` grows
/// past three. At exactly three runs that robustness is not there to be had —
/// `mean_x` is 1 and the `ȳ` terms cancel, so the fitted change reduces
/// *exactly* to `last − first` and the middle run contributes nothing but its
/// share of the median. That degeneracy is why the percentage test is not the
/// whole test.
///
/// Two conditions, both necessary. The fitted change must clear
/// [`TREND_MAX_DRIFT_PCT`] of the median — it has to be *large* — and it must
/// also exceed [`TREND_MIN_RESID_MULTIPLE`] times the RMS of the residuals about
/// the fitted line — it has to be large *relative to the cell's own scatter*. A
/// percentage alone cannot tell a ramp from a zig-zag at three runs, where the
/// fit degenerates to `last − first`; the residual test is what supplies the
/// missing noise anchoring.
///
/// The residual RMS divides by `n`, not by the regression's `n − 2` degrees of
/// freedom. With the default three runs `n − 2` is one, and the resulting
/// estimate swings by a factor of `√3` for no gain: the quantity wanted here is
/// "how far do these points sit off the line", not an unbiased variance.
///
/// At the two-run minimum the line passes through both points, so the residual
/// RMS is exactly zero and the percentage floor governs alone. That is the right
/// answer rather than a gap: with two points there is no scatter to measure, and
/// anything else would be inventing one.
///
/// `None` when there is no trend worth refusing over: fewer than two runs (no
/// order to speak of), a zero median (no scale to express the change against),
/// or a fitted change that fails either condition.
pub(crate) fn detect_drift(values_in_order: &[f64], median: f64) -> Option<f64> {
    let n = values_in_order.len();
    if n < 2 || median == 0.0 {
        return None;
    }
    #[allow(
        clippy::cast_precision_loss,
        reason = "n is the run count — single digits in every real invocation"
    )]
    let n_f = n as f64;
    let mean_x = (n_f - 1.0) / 2.0;
    let mean_y = values_in_order.iter().sum::<f64>() / n_f;
    let (mut sxy, mut sxx) = (0.0_f64, 0.0_f64);
    for (i, y) in values_in_order.iter().enumerate() {
        #[allow(
            clippy::cast_precision_loss,
            reason = "i indexes the run count — single digits in every real invocation"
        )]
        let dx = i as f64 - mean_x;
        sxy += dx * (y - mean_y);
        sxx += dx * dx;
    }
    if sxx == 0.0 {
        return None;
    }
    let slope = sxy / sxx;
    // Change across the whole sequence: first measured run to last.
    let fitted_change = slope * (n_f - 1.0);
    let pct_of_median = fitted_change / median.abs() * 100.0;
    if pct_of_median.abs() <= TREND_MAX_DRIFT_PCT {
        return None;
    }
    // Scatter about the fitted line. A change that does not dominate this is a
    // sample that happens to end higher than it started, not a trend.
    let mut sq_resid = 0.0_f64;
    for (i, y) in values_in_order.iter().enumerate() {
        #[allow(
            clippy::cast_precision_loss,
            reason = "i indexes the run count — single digits in every real invocation"
        )]
        let dx = i as f64 - mean_x;
        let resid = y - (mean_y + slope * dx);
        sq_resid += resid * resid;
    }
    let resid_rms = (sq_resid / n_f).sqrt();
    (fitted_change.abs() > TREND_MIN_RESID_MULTIPLE * resid_rms).then_some(pct_of_median)
}

/// Why a metric cannot be summarised as one cell, or `None` when it settled.
///
/// Two shapes of not-settling, checked in order of how specific the diagnosis
/// is. A trend names *which way* the cell was moving and has a remedy (more
/// warmup); a wide range only says the runs never converged, and its remedy is
/// the host, not the run count. Reported as a string rather than raised here so
/// the caller can name every metric that failed instead of only the first.
fn settle_refusal(name: &str, values_in_order: &[f64], spread: &Spread) -> Option<String> {
    let ordered: Vec<String> = values_in_order.iter().map(|v| format!("{v:.3}")).collect();
    let ordered = ordered.join(" → ");
    if let Some(pct) = detect_drift(values_in_order, spread.median) {
        let direction = if pct > 0.0 { "rose" } else { "fell" };
        return Some(format!(
            "{name} {direction} {:.1}% from the first measured run to the last ({ordered}, in \
             run order): this is a trend, not a spread, and its median is not a measurement. \
             The cell had not reached a steady state — raise --warmup until consecutive runs \
             agree, or measure a cell that settles",
            pct.abs(),
        ));
    }
    let range_pct = spread.range_pct();
    if range_pct > SETTLED_MAX_RANGE_PCT {
        return Some(format!(
            "{name} spanned {range_pct:.1}% of its median across the measured runs ({ordered}, \
             in run order), over the {SETTLED_MAX_RANGE_PCT:.0}% ceiling: the runs never \
             converged on a value, so the median is the middle of a scatter rather than a \
             measurement. Quiet the host, or measure a cell that settles"
        ));
    }
    None
}

// ---------------------------------------------------------------------------
// KV-byte reading — three states, not two
// ---------------------------------------------------------------------------

/// Turn a verdict into a usable byte count or the reason there is none.
fn kv_bytes_or_reason(verdict: KvBytesVerdict, arch_class: &str) -> anyhow::Result<u64> {
    match verdict {
        KvBytesVerdict::Reported(n) => Ok(n),
        KvBytesVerdict::Unreported => Err(anyhow::anyhow!(
            "generation produced no KV-cache byte count on arch {arch_class}: the store \
             sequence did not advance, so the readable figure belongs to an earlier \
             generation or is the unset initialiser. Refusing to report it as this run's \
             measurement"
        )),
        KvBytesVerdict::ReportedZero => Err(anyhow::anyhow!(
            "generation on arch {arch_class} reported a KV cache of 0 bytes after a real \
             prefill — the byte accounting is wrong, not the cache. Refusing to report a \
             zero-byte KV measurement"
        )),
    }
}

// ---------------------------------------------------------------------------
// Prompt-cache guard — a TTFT is only a TTFT when prefill actually ran
// ---------------------------------------------------------------------------

/// Check that one measured generation actually ran prefill.
///
/// `stats` is the arch's prompt-cache counters read *after* the generation, or
/// `None` when the arch has no cache to report. Taken as the named
/// [`CacheStats`] rather than a `(hits, misses)` pair: with a tuple, swapping
/// the two at the call site type-checks, passes every test that builds tuples
/// directly, and silently inverts the guard.
///
/// Three conditions, all necessary:
///
/// - **`hits == 0`.** A hit means the post-prefill KV snapshot was replayed, so
///   the run's TTFT is a cache-replay time, not a time-to-first-token — a
///   number that is small, stable, and wrong.
/// - **`ssd_hits == 0`.** A RAM miss that the SSD tier served is *also* a
///   skipped prefill, and it does not show up in `hits`: `hydrate_from_ssd`
///   bumps its own counter after `find_best_prefix` has already recorded the
///   miss. Clearing the RAM slots does not detach the SSD source, so the
///   emptied-cache guarantee that makes `hits == 0, misses == 1` mean "a real
///   prefill ran" holds only while nothing hydrates. Reading the run's TTFT off
///   a `.kvb` reconstruction is the same defect wearing a different counter.
/// - **`misses >= 1`.** The cache was consulted and did not serve this run, so
///   a prefill genuinely happened. Counters that report nothing certify
///   nothing.
///
/// Absolute counters rather than a before/after delta, deliberately: the
/// per-run `clear_prompt_cache` resets the counters along with the slots, and a
/// delta across a reset reads as "no activity". The rule holds either way —
/// under stable counters a repeat that was served shows `hits > 0`, and after a
/// clear the run shows `hits == 0, misses == 1`.
pub(crate) fn assert_prefill_measured(stats: Option<&CacheStats>) -> anyhow::Result<()> {
    let Some(&CacheStats {
        hits,
        misses,
        ssd_hits,
        ..
    }) = stats
    else {
        return Err(anyhow::anyhow!(
            "measured run reported no prompt-cache stats, so nothing certifies that it \
             performed a prefill rather than replaying a snapshot. Refusing to report its \
             TTFT"
        ));
    };
    if hits > 0 {
        return Err(anyhow::anyhow!(
            "measured run was served from the prompt cache ({hits} hit(s), {misses} miss(es)): \
             prefill was skipped, so its TTFT and prefill throughput are cache-replay times, \
             not measurements. Refusing to report them"
        ));
    }
    if ssd_hits > 0 {
        return Err(anyhow::anyhow!(
            "measured run was served from the SSD KV tier ({ssd_hits} hydrate(s)): the RAM \
             cache missed, but the blocks were reconstructed from a .kvb file instead of \
             being prefilled, so its TTFT is a hydrate time. Emptying the RAM slots does not \
             detach the SSD source. Refusing to report it"
        ));
    }
    if misses == 0 {
        return Err(anyhow::anyhow!(
            "measured run recorded no prompt-cache miss: the cache was never consulted, so \
             the counters do not certify that this run performed a fresh prefill and its \
             TTFT cannot be attributed to it"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Machine load — a contended host is not a measurement condition
// ---------------------------------------------------------------------------

/// Parse the 1-minute figure out of `sysctl -n vm.loadavg` output, which is
/// `{ <1m> <5m> <15m> }`.
///
/// `None` when the text does not have that shape — an unparseable reading is
/// reported as "no reading", never as a load of zero, which would silently
/// certify a busy host as quiet.
pub(crate) fn parse_loadavg_1m(sysctl_out: &str) -> Option<f64> {
    sysctl_out
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split_whitespace()
        .next()?
        .parse::<f64>()
        .ok()
}

/// 1-minute load average, or `None` when the OS declines to report one.
///
/// Shells out rather than calling `getloadavg`: this crate denies `unsafe`, and
/// the CLI already reads process figures through `ps` the same way.
fn load_average_1m() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_loadavg_1m(&String::from_utf8_lossy(&out.stdout))
}

/// Whether the host looked busy enough for the measurement to be suspect.
///
/// The threshold is the CPU count: at or above it, runnable work outnumbers
/// cores and the decode loop is competing for them. This does not abort the
/// run — the operator may knowingly be measuring under load — but it is
/// reported next to the numbers so a contended figure is never mistaken for a
/// quiet-machine one.
pub(crate) fn load_is_contended(load_1m: f64, cpus: usize) -> bool {
    load_1m >= cpus as f64
}

// ---------------------------------------------------------------------------
// One measured run
// ---------------------------------------------------------------------------

/// FNV-1a-64 over a run's token ids.
///
/// Stable across processes and releases (unlike `DefaultHasher`), so two
/// invocations of `bench` on the same cell can be compared by eye from the
/// printed digest.
pub(crate) fn token_stream_digest(token_ids: impl IntoIterator<Item = u32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in token_ids {
        for b in id.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// Everything one generation contributes to the summary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunSample {
    /// Prefill through first token, milliseconds.
    pub ttft_ms: f64,
    /// Steady-state decode rate over tokens 2..N, tokens/second.
    pub decode_tps: f64,
    /// Prefill throughput, prompt tokens/second. `None` when the run produced
    /// nothing to divide by — never a zero standing in for one.
    pub prefill_tps: Option<f64>,
    /// Median inter-token latency within the run, milliseconds.
    pub itl_p50_ms: f64,
    /// 99th-percentile inter-token latency within the run, milliseconds.
    pub itl_p99_ms: f64,
    /// Filled-prefix KV cache bytes after decode.
    pub kv_cache_bytes: u64,
    /// Tokens actually generated.
    pub n_generated: usize,
    /// FNV-1a-64 over the run's token ids. Every run in a cell decodes the same
    /// prompt at temperature 0, so every run must produce the same digest.
    pub token_digest: u64,
}

/// Derive a run's metrics from the per-token arrival times.
///
/// `arrivals_s` are the elapsed seconds, relative to generation start, at which
/// each token became available — one entry per generated token, in order. The
/// first entry is the TTFT; the gaps between consecutive entries are the
/// inter-token latencies.
///
/// `None` when fewer than two tokens arrived: with one token there is no
/// inter-token interval and no steady-state rate, and reporting the combined
/// prefill+decode number in their place would label a prefill-dominated figure
/// as a decode rate.
pub(crate) fn sample_from_arrivals(
    arrivals_s: &[f64],
    prompt_tokens: usize,
    kv_cache_bytes: u64,
    token_digest: u64,
) -> Option<RunSample> {
    if arrivals_s.len() < 2 {
        return None;
    }
    let first = *arrivals_s.first()?;
    let last = *arrivals_s.last()?;

    let mut itl_ms: Vec<f64> = arrivals_s
        .windows(2)
        .filter_map(|w| Some((w.get(1)? - w.first()?) * 1000.0))
        .collect();
    itl_ms.sort_by(f64::total_cmp);

    let decode_window_s = last - first;
    let decode_tps = if decode_window_s > 0.0 {
        (arrivals_s.len() as f64 - 1.0) / decode_window_s
    } else {
        return None;
    };
    // `None`, not `0.0`: a zero here would be summarised and printed exactly
    // like a measured throughput of zero — the one thing this file exists to
    // prevent.
    #[allow(
        clippy::cast_precision_loss,
        reason = "prompt lengths are far below 2^53; f64 is exact over the whole range"
    )]
    let prefill_tps = (first > 0.0 && prompt_tokens > 0).then(|| prompt_tokens as f64 / first);

    Some(RunSample {
        ttft_ms: first * 1000.0,
        decode_tps,
        prefill_tps,
        itl_p50_ms: percentile_sorted(&itl_ms, 0.50)?,
        itl_p99_ms: percentile_sorted(&itl_ms, 0.99)?,
        kv_cache_bytes,
        n_generated: arrivals_s.len(),
        token_digest,
    })
}

/// Arguments for one `rmlx bench` invocation.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — the argument bundle main.rs builds for the single call site"
)]
#[derive(Debug)]
pub(crate) struct BenchArgs {
    /// Model snapshot directory.
    pub model: PathBuf,
    /// Prompt file (plain text or chat-JSON fixture).
    pub prompt: PathBuf,
    /// Short label for the prompt in the summary.
    pub prompt_label: String,
    /// Inference device.
    pub device: Device,
    /// Tokens to generate per run.
    pub max_tokens: u32,
    /// Measured runs (>= `MIN_RUNS`).
    pub runs: u32,
    /// Discarded warmup runs before the measured ones.
    pub warmup: u32,
    /// Resolved KV-cache codec.
    pub kv_quant: rmlx_kv_quant::KvQuant,
    /// KV ring capacity override, when given.
    pub max_ctx: Option<i32>,
    /// Cap on tokenized prompt length.
    pub max_prompt_tokens: usize,
    /// Whether `--max-prompt-tokens` was passed explicitly.
    pub cap_is_explicit: bool,
    /// Opt into truncating an over-cap prompt on GPU.
    pub allow_truncate: bool,
    /// Emit the summary as one JSON object instead of a table.
    pub json: bool,
}

/// Run one generation and return its sample, or the reason there is none.
fn measure_one(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    args: &BenchArgs,
) -> anyhow::Result<RunSample> {
    let sampler_cfg = rmlx_models::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut rng = rmlx_models::Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = rmlx_models::PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    // Empty the prompt cache so this generation is a guaranteed miss and runs a
    // real prefill. Done here, before the clock starts, so the clearing is not
    // charged to the TTFT it protects.
    model.clear_prompt_cache();

    let kv_before = model.kv_cache_bytes_sample();

    // Per-token arrival stamps. Reserved once, so the callback never allocates
    // and the timing it records is not the timing of its own bookkeeping.
    let mut arrivals_s: Vec<f64> = Vec::with_capacity(args.max_tokens as usize);
    let t0 = Instant::now();
    let mut on_token = |_step: &rmlx_models::ProbeStep| -> Option<u32> {
        arrivals_s.push(t0.elapsed().as_secs_f64());
        None
    };

    let steps = model
        .generate_greedy(
            tokenizer,
            prompt_ids,
            args.max_tokens as usize,
            args.device,
            Some(args.kv_quant),
            args.max_ctx,
            BENCH_PROMPT_CACHE_SLOTS,
            &[], // no EOS stop: every run must decode the same token count
            &mut on_token,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .map_err(|e| anyhow::anyhow!("generate_greedy: {e}"))?;

    assert_prefill_measured(model.cache_stats().as_ref())?;

    let kv_cache_bytes = kv_bytes_or_reason(
        classify_kv_bytes(kv_before, model.kv_cache_bytes_sample()),
        model.arch_class(),
    )?;

    // The callback fires once per emitted token, so a mismatch means the two
    // disagree about how many tokens the run produced — the arrival stamps
    // could not then be attributed to the returned tokens.
    if arrivals_s.len() != steps.len() {
        return Err(anyhow::anyhow!(
            "per-token callback fired {} time(s) but the run returned {} token(s): the \
             arrival timestamps cannot be matched to the generated tokens, so the \
             inter-token latencies are not this run's",
            arrivals_s.len(),
            steps.len()
        ));
    }

    let token_digest = token_stream_digest(steps.iter().map(|s| s.token_id));

    sample_from_arrivals(&arrivals_s, prompt_ids.len(), kv_cache_bytes, token_digest).ok_or_else(
        || {
            anyhow::anyhow!(
                "run generated {} token(s); at least 2 are needed for an inter-token latency \
             and a steady-state decode rate. Raise --max-tokens",
                arrivals_s.len()
            )
        },
    )
}

// ---------------------------------------------------------------------------
// Metrics — one enumeration, one gate decision each
// ---------------------------------------------------------------------------

/// What the summary requires of a metric before it will report it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gate {
    /// The cell must have settled in this metric: no run-to-run trend, and a
    /// range narrow enough to be noise around a centre.
    Settled,
    /// Reported with its spread, never gated. Each metric that opts out states
    /// why on [`Metric::gate`].
    Ungated,
}

/// Every metric one bench cell reports.
///
/// The point of the enum is [`Metric::gate`]: whether a metric is checked for
/// settling used to be positional — a call that was made for some metrics and
/// simply absent for others — so a metric added later inherited whichever
/// treatment its author happened to copy. Here the exhaustive match makes the
/// choice mandatory: a new variant does not compile until it names its gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Metric {
    TtftMs,
    DecodeTps,
    PrefillTps,
    ItlP50Ms,
    ItlP99Ms,
    KvCacheBytes,
}

impl Metric {
    /// Reporting order, which is also summary-row order.
    pub(crate) const ALL: [Self; 6] = [
        Self::TtftMs,
        Self::ItlP50Ms,
        Self::ItlP99Ms,
        Self::DecodeTps,
        Self::PrefillTps,
        Self::KvCacheBytes,
    ];

    /// Label used in the summary and in every refusal message.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::TtftMs => "ttft_ms",
            Self::DecodeTps => "decode_tps",
            Self::PrefillTps => "prefill_tps",
            Self::ItlP50Ms => "itl_p50_ms",
            Self::ItlP99Ms => "itl_p99_ms",
            Self::KvCacheBytes => "kv_cache_bytes",
        }
    }

    /// Whether the cell has to have settled in this metric.
    #[allow(
        clippy::match_same_arms,
        reason = "the two ungated metrics opt out for unrelated reasons and each states its own; \
                  merging the arms would drop one of the two justifications and leave the next \
                  metric added inheriting whichever it landed beside"
    )]
    pub(crate) const fn gate(self) -> Gate {
        match self {
            // The three robust per-run quantities plus the byte total. These
            // are what a decision gets made on, so a cell that has not settled
            // in any of them cannot support one.
            Self::TtftMs | Self::DecodeTps | Self::ItlP50Ms | Self::KvCacheBytes => Gate::Settled,

            // Nearest-rank p99 over a 128-token run is the second-largest
            // inter-token gap: an extreme-order statistic whose run-to-run
            // movement is dominated by whether that one run happened to hit a
            // stall, not by whether the cell has settled. Measured gemma-4-e2b
            // at 4k: ttft, decode TPS and ITL p50 all moved 5-6% across three
            // runs while p99 moved 26%. Gating it would abort cells whose
            // measurements are fine; its spread is still printed, so the tail
            // stays visible.
            Self::ItlP99Ms => Gate::Ungated,

            // An exact deterministic reciprocal of `ttft_ms` — the prompt is
            // identical in every run of a cell, so `prompt_tokens / ttft` adds
            // no independent evidence. Gating it would test `ttft_ms` twice
            // under a nonlinear transform, and the two could disagree purely on
            // where the middle run landed. `ttft_ms` carries the gate.
            Self::PrefillTps => Gate::Ungated,
        }
    }

    /// This metric's per-run values in collection order.
    ///
    /// `None` when a run could not produce the quantity at all — only
    /// [`Metric::PrefillTps`] can be absent, and a cell where only some runs
    /// have one is not a cell.
    #[allow(
        clippy::cast_precision_loss,
        reason = "KV byte totals are far below 2^53; f64 is exact over the whole range"
    )]
    fn values(self, samples: &[RunSample]) -> Option<Vec<f64>> {
        match self {
            Self::TtftMs => Some(samples.iter().map(|s| s.ttft_ms).collect()),
            Self::DecodeTps => Some(samples.iter().map(|s| s.decode_tps).collect()),
            Self::PrefillTps => samples.iter().map(|s| s.prefill_tps).collect(),
            Self::ItlP50Ms => Some(samples.iter().map(|s| s.itl_p50_ms).collect()),
            Self::ItlP99Ms => Some(samples.iter().map(|s| s.itl_p99_ms).collect()),
            Self::KvCacheBytes => Some(samples.iter().map(|s| s.kv_cache_bytes as f64).collect()),
        }
    }
}

/// Summary of one bench cell.
#[derive(Debug)]
struct BenchSummary {
    ttft_ms: Spread,
    decode_tps: Spread,
    /// `None` when any run could not produce a prefill throughput. A cell where
    /// only some runs have one is not a cell.
    prefill_tps: Option<Spread>,
    itl_p50_ms: Spread,
    itl_p99_ms: Spread,
    kv_bytes: Spread,
    /// The one token-stream digest every run agreed on.
    token_digest: u64,
}

/// Fold the measured runs into per-metric spreads, refusing any metric in which
/// the cell did not settle.
///
/// Errors when `samples` is empty, and when any [`Gate::Settled`] metric
/// trended or scattered. Every metric is summarised from the same runs, so
/// either all of them exist or none do — there is no partial summary.
///
/// Every gated metric is checked before anything is refused, so the error names
/// *all* of them. Refusing at the first one hides how much of the cell moved,
/// which is the difference between "one metric wobbled" and "the whole run was
/// contended".
fn summarize(samples: &[RunSample]) -> anyhow::Result<BenchSummary> {
    let missing = || anyhow::anyhow!("no measured runs completed — nothing to summarise");

    let mut spreads: Vec<(Metric, Spread)> = Vec::with_capacity(Metric::ALL.len());
    let mut refusals: Vec<String> = Vec::new();
    for metric in Metric::ALL {
        let Some(in_order) = metric.values(samples) else {
            continue;
        };
        let spread = Spread::of(&in_order).ok_or_else(missing)?;
        if metric.gate() == Gate::Settled {
            refusals.extend(settle_refusal(metric.name(), &in_order, &spread));
        }
        spreads.push((metric, spread));
    }
    if !refusals.is_empty() {
        return Err(anyhow::anyhow!(
            "this cell did not settle in {} of its {} gated metric(s), so it has no median to \
             report:\n  - {}",
            refusals.len(),
            Metric::ALL
                .iter()
                .filter(|m| m.gate() == Gate::Settled)
                .count(),
            refusals.join("\n  - ")
        ));
    }

    let get = |metric: Metric| spreads.iter().find(|(m, _)| *m == metric).map(|&(_, s)| s);
    Ok(BenchSummary {
        ttft_ms: get(Metric::TtftMs).ok_or_else(missing)?,
        decode_tps: get(Metric::DecodeTps).ok_or_else(missing)?,
        prefill_tps: get(Metric::PrefillTps),
        itl_p50_ms: get(Metric::ItlP50Ms).ok_or_else(missing)?,
        itl_p99_ms: get(Metric::ItlP99Ms).ok_or_else(missing)?,
        kv_bytes: get(Metric::KvCacheBytes).ok_or_else(missing)?,
        token_digest: samples.first().ok_or_else(missing)?.token_digest,
    })
}

/// Tokenize the prompt and apply the length cap.
fn prepare_prompt(args: &BenchArgs, tokenizer: &tokenizers::Tokenizer) -> anyhow::Result<Vec<u32>> {
    let prompt_text = std::fs::read_to_string(&args.prompt)
        .map_err(|e| anyhow::anyhow!("cannot read prompt file {}: {e}", args.prompt.display()))?;
    let (mut prompt_ids, template_used) =
        tokenize_prompt_text(&args.model, tokenizer, &prompt_text)?;
    let effective_len = super::baseline::resolve_prompt_truncation(
        prompt_ids.len(),
        args.max_prompt_tokens,
        args.device,
        args.cap_is_explicit,
        args.allow_truncate,
    )?;
    prompt_ids.truncate(effective_len);
    info!(
        model = %args.model.display(),
        device = ?args.device,
        kv_quant = %args.kv_quant,
        prompt_tokens = prompt_ids.len(),
        max_tokens = args.max_tokens,
        runs = args.runs,
        warmup = args.warmup,
        template_used,
        "bench: starting"
    );
    Ok(prompt_ids)
}

/// Check that every run of the cell decoded the same tokens.
///
/// Every run in a cell feeds the same prompt to the same model at temperature 0
/// with a fixed seed, so every run must emit a byte-identical token stream. A
/// cell whose runs disagree is not measuring one thing N times, and its timings
/// describe N different generations.
///
/// This is also the only output check `bench` performs, and it is load-bearing
/// beyond reproducibility: a KV cache that silently stops being written decodes
/// *faster* while producing wrong tokens, so a timing-only instrument is biased
/// toward accepting exactly that defect.
///
/// The reference is the **most common** digest, not the first run's. Anchoring
/// on run 1 makes a cold-start defect — the case a repeated-run instrument is
/// most likely to meet — report every *later* run as the deviant one, pointing
/// the operator away from the run that actually misbehaved. The message lists
/// every run with its digest either way, so a two-run cell with no majority is
/// still fully described.
fn assert_one_token_stream(runs: &[(String, u64)]) -> anyhow::Result<()> {
    let mut counts: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for &(_, digest) in runs {
        *counts.entry(digest).or_insert(0) += 1;
    }
    if counts.len() <= 1 {
        return Ok(());
    }
    // Ties broken by the smaller digest, so the message is deterministic.
    let (majority, majority_n) = counts
        .into_iter()
        .max_by_key(|&(digest, n)| (n, std::cmp::Reverse(digest)))
        .ok_or_else(|| anyhow::anyhow!("no runs to compare token streams across"))?;
    let listing: Vec<String> = runs
        .iter()
        .map(|(label, digest)| {
            let mark = if *digest == majority {
                ""
            } else {
                "  <-- differs"
            };
            format!("{label}: {digest:#018x}{mark}")
        })
        .collect();
    Err(anyhow::anyhow!(
        "the runs of this cell decoded different token streams, so they are not repeats of one \
         measurement — and a cache that stops being written decodes faster while producing \
         wrong tokens. Most common digest {majority:#018x} ({majority_n} of {} run(s)):\n  {}\n\
         Refusing to summarise them as one cell",
        runs.len(),
        listing.join("\n  ")
    ))
}

/// Run the warmup runs (discarded) then the measured ones.
///
/// A failure in any run aborts the whole invocation: a summary built from the
/// runs that happened to succeed would be a different measurement than the one
/// requested, reported under the requested one's label.
fn collect_samples(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    args: &BenchArgs,
) -> anyhow::Result<Vec<RunSample>> {
    // The warmup runs are discarded as measurements, but their tokens are still
    // evidence: a warmup that decodes something else than the measured runs is
    // the same defect and costs nothing to catch. Digests are compared once, at
    // the end, over every run — see `assert_one_token_stream`.
    let mut digests: Vec<(String, u64)> = Vec::with_capacity((args.warmup + args.runs) as usize);

    for i in 0..args.warmup {
        info!(run = i + 1, of = args.warmup, "bench: warmup run");
        let s = measure_one(model, tokenizer, prompt_ids, args)
            .map_err(|e| anyhow::anyhow!("warmup run {} failed: {e}", i + 1))?;
        digests.push((format!("warmup run {}", i + 1), s.token_digest));
    }

    let mut samples: Vec<RunSample> = Vec::with_capacity(args.runs as usize);
    for i in 0..args.runs {
        let s = measure_one(model, tokenizer, prompt_ids, args)
            .map_err(|e| anyhow::anyhow!("measured run {} failed: {e}", i + 1))?;
        digests.push((format!("measured run {}", i + 1), s.token_digest));
        // Every reported metric appears here, so a run that the summary later
        // refuses is still recoverable from the log — including `prefill_tps`,
        // which is otherwise only ever printed as part of a summary that a
        // refusal discards.
        info!(
            run = i + 1,
            of = args.runs,
            ttft_ms = s.ttft_ms,
            decode_tps = s.decode_tps,
            prefill_tps = s.prefill_tps,
            itl_p50_ms = s.itl_p50_ms,
            itl_p99_ms = s.itl_p99_ms,
            kv_cache_bytes = s.kv_cache_bytes,
            token_digest = format!("{:#018x}", s.token_digest),
            "bench: measured run"
        );
        samples.push(s);
    }
    assert_one_token_stream(&digests)?;
    Ok(samples)
}

/// Execute `rmlx bench`.
#[allow(
    clippy::needless_pass_by_value,
    reason = "owns the argument bundle for the whole invocation; main.rs builds it and hands it over"
)]
pub(crate) fn run_bench(args: BenchArgs) -> anyhow::Result<()> {
    if args.runs < MIN_RUNS {
        return Err(anyhow::anyhow!(
            "--runs must be at least {MIN_RUNS}: a single measurement has no observable \
             run-to-run spread, and this instrument does not report a central value without one"
        ));
    }

    let tokenizer = tokenizers::Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("cannot load tokenizer.json: {e}"))?;
    let prompt_ids = prepare_prompt(&args, &tokenizer)?;

    let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let load_before = load_average_1m();
    if load_before.is_some_and(|l| load_is_contended(l, cpus)) {
        warn!(
            load_1m = load_before,
            cpus, "host is busy — bench numbers are contended, not quiet-machine figures"
        );
    }

    let load_start = Instant::now();
    let model = arch::load_model(&args.model, args.device, &arch::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("arch::load_model: {e}"))?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    info!(load_ms, arch = model.arch_class(), "bench: model loaded");

    let samples = collect_samples(&model, &tokenizer, &prompt_ids, &args)?;
    let summary = summarize(&samples)?;

    let load_after = load_average_1m();
    let ctx = ReportCtx {
        model_name: args
            .model
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)"),
        arch_class: model.arch_class(),
        args: &args,
        prompt_tokens: prompt_ids.len(),
        gen_tokens: samples.first().map_or(0, |s| s.n_generated),
        load_ms,
        load_before,
        load_after,
        cpus,
        contended: [load_before, load_after]
            .into_iter()
            .flatten()
            .any(|l| load_is_contended(l, cpus)),
    };

    if args.json {
        emit_json(&ctx, &summary, &samples)?;
    } else {
        emit_table(&ctx, &summary);
    }
    Ok(())
}

/// Run context the summary needs alongside the metrics themselves.
struct ReportCtx<'a> {
    model_name: &'a str,
    arch_class: &'a str,
    args: &'a BenchArgs,
    prompt_tokens: usize,
    gen_tokens: usize,
    load_ms: f64,
    load_before: Option<f64>,
    load_after: Option<f64>,
    cpus: usize,
    contended: bool,
}

fn spread_json(s: &Spread) -> serde_json::Value {
    serde_json::json!({
        "median": s.median,
        "min": s.min,
        "max": s.max,
        "range_pct": s.range_pct(),
        "n": s.n,
    })
}

/// Machine-readable summary. Carries every individual run alongside the
/// per-metric spreads, so a consumer can re-derive the spread rather than
/// having to trust the summary.
fn emit_json(ctx: &ReportCtx<'_>, s: &BenchSummary, samples: &[RunSample]) -> anyhow::Result<()> {
    let doc = serde_json::json!({
        "model": ctx.model_name,
        "arch": ctx.arch_class,
        "kv_quant": ctx.args.kv_quant.to_string(),
        "prompt": ctx.args.prompt_label,
        "prompt_tokens": ctx.prompt_tokens,
        "gen_tokens": ctx.gen_tokens,
        "runs": ctx.args.runs,
        "warmup": ctx.args.warmup,
        "load_ms": ctx.load_ms,
        "load_1m_before": ctx.load_before,
        "load_1m_after": ctx.load_after,
        "cpus": ctx.cpus,
        "contended": ctx.contended,
        "token_digest": format!("{:#018x}", s.token_digest),
        "metrics": {
            "ttft_ms": spread_json(&s.ttft_ms),
            "decode_tps": spread_json(&s.decode_tps),
            "prefill_tps": s.prefill_tps.as_ref().map(spread_json),
            "itl_p50_ms": spread_json(&s.itl_p50_ms),
            "itl_p99_ms": spread_json(&s.itl_p99_ms),
            "kv_cache_bytes": spread_json(&s.kv_bytes),
        },
        "runs_detail": samples.iter().map(|r| serde_json::json!({
            "ttft_ms": r.ttft_ms,
            "decode_tps": r.decode_tps,
            "prefill_tps": r.prefill_tps,
            "itl_p50_ms": r.itl_p50_ms,
            "itl_p99_ms": r.itl_p99_ms,
            "kv_cache_bytes": r.kv_cache_bytes,
            "token_digest": format!("{:#018x}", r.token_digest),
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&doc)?);
    Ok(())
}

/// Human-readable summary. Every row carries min/max so no reader sees a
/// central value without its spread.
fn emit_table(ctx: &ReportCtx<'_>, s: &BenchSummary) {
    println!(
        "bench: model={} arch={} kv_quant={} prompt={} prompt_tokens={} gen_tokens={} \
         runs={} warmup={} load={:.0}ms",
        ctx.model_name,
        ctx.arch_class,
        ctx.args.kv_quant,
        ctx.args.prompt_label,
        ctx.prompt_tokens,
        ctx.gen_tokens,
        ctx.args.runs,
        ctx.args.warmup,
        ctx.load_ms
    );
    println!(
        "{:<16} {:>14} {:>14} {:>14} {:>9}",
        "metric", "median", "min", "max", "range%"
    );
    let row = |name: &str, v: &Spread, prec: usize| {
        println!(
            "{name:<16} {:>14.prec$} {:>14.prec$} {:>14.prec$} {:>8.1}%",
            v.median,
            v.min,
            v.max,
            v.range_pct(),
            prec = prec
        );
    };
    row("ttft_ms", &s.ttft_ms, 2);
    row("itl_p50_ms", &s.itl_p50_ms, 3);
    row("itl_p99_ms", &s.itl_p99_ms, 3);
    row("decode_tps", &s.decode_tps, 3);
    match s.prefill_tps.as_ref() {
        Some(v) => row("prefill_tps", v, 1),
        // No run produced a prefill throughput. Say so rather than print a
        // zero that reads as a measured throughput of zero.
        None => println!("{:<16} {:>14}", "prefill_tps", "n/a"),
    }
    row("kv_cache_bytes", &s.kv_bytes, 0);
    println!(
        "tokens: digest={:#018x} (identical across every run)",
        s.token_digest
    );

    let fmt_load = |l: Option<f64>| l.map_or_else(|| "n/a".to_owned(), |v| format!("{v:.2}"));
    println!(
        "host: cpus={} load_1m={}→{}{}",
        ctx.cpus,
        fmt_load(ctx.load_before),
        fmt_load(ctx.load_after),
        if ctx.contended {
            "  CONTENDED — measured while the host was busy, treat as a lower bound"
        } else {
            ""
        }
    );
}

/// Resolve a prompt path the same way `rmlx baseline` does.
pub(crate) fn resolve_prompt(
    prompts_root: &Path,
    prompt: PathBuf,
    prompt_tokens: Option<u32>,
) -> anyhow::Result<(PathBuf, String)> {
    if let Some(n) = prompt_tokens {
        return super::baseline::resolve_prompt_tokens_file(prompts_root, n);
    }
    let label = prompt
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown")
        .to_owned();
    Ok((prompt, label))
}

#[cfg(test)]
#[path = "bench_tests.rs"]
mod bench_tests;
