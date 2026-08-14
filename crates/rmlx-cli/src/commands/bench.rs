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
//!    2 so there is always a spread to report. A metric that *trends* across
//!    those runs has no central value at all, and is refused rather than
//!    summarised — see [`detect_drift`].
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
/// before every generation: the cache is emptied, so the next request is a
/// guaranteed miss and a real prefill. Asking for *zero* slots would not do it
/// — capacity is clamped to a minimum of one, so a "zero-slot" cache still
/// stores and can still serve a snapshot.
///
/// `assert_prefill_measured` re-checks the outcome per run rather than trusting
/// either the constant or the clear.
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
const TREND_MAX_DRIFT_PCT: f64 = 10.0;

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
    /// Takes the sample by value and sorts it in place: the caller has already
    /// built the vector, and a borrowing signature would only copy it again.
    pub(crate) fn of(values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let mut sorted = values;
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
/// still a ramp, and would escape a strict-monotonic check the moment `--runs`
/// grew past three.
///
/// At the default three runs, a sample whose spread is very wide fits a slope
/// in *whatever* order it arrives — with a 40%-wide sample, every permutation
/// clears the threshold. That is not a flaw to tune away: three runs cannot
/// separate a ramp from that much noise, and a cell that noisy has no
/// trustworthy median either. Both are refused, which is the safe direction;
/// the error prints the values in run order so the operator can see which it
/// was.
///
/// `None` when there is no trend worth refusing over: fewer than two runs (no
/// order to speak of), a zero median (no scale to express the change against),
/// or a fitted change within [`TREND_MAX_DRIFT_PCT`].
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
    // Slope per run, scaled across the whole sequence: first run to last.
    let pct_of_median = sxy / sxx * (n_f - 1.0) / median.abs() * 100.0;
    (pct_of_median.abs() > TREND_MAX_DRIFT_PCT).then_some(pct_of_median)
}

/// Refuse to summarise a metric that trended across its own runs.
fn assert_no_drift(name: &str, values_in_order: &[f64], median: f64) -> anyhow::Result<()> {
    let Some(pct) = detect_drift(values_in_order, median) else {
        return Ok(());
    };
    let direction = if pct > 0.0 { "rose" } else { "fell" };
    let ordered: Vec<String> = values_in_order.iter().map(|v| format!("{v:.3}")).collect();
    Err(anyhow::anyhow!(
        "{name} {direction} {:.1}% from the first measured run to the last ({}, in run order): \
         this is a trend, not a spread, and its median is not a measurement. The cell had not \
         reached a steady state — raise --warmup until consecutive runs agree, or measure a \
         cell that settles",
        pct.abs(),
        ordered.join(" → ")
    ))
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
/// Two conditions, both necessary:
///
/// - **`hits == 0`.** A hit means the post-prefill KV snapshot was replayed, so
///   the run's TTFT is a cache-replay time, not a time-to-first-token — a
///   number that is small, stable, and wrong.
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
    let Some(&CacheStats { hits, misses, .. }) = stats else {
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

/// Summary of one bench cell.
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

/// Fold the measured runs into per-metric spreads, refusing any metric that
/// trended across the runs instead of settling.
///
/// Errors when `samples` is empty, and when any checked metric drifted. Every
/// metric is summarised from the same runs, so either all of them exist or none
/// do — there is no partial summary.
fn summarize(samples: &[RunSample]) -> anyhow::Result<BenchSummary> {
    let missing = || anyhow::anyhow!("no measured runs completed — nothing to summarise");

    // Each metric is checked for a trend *in collection order* before its
    // order-blind spread is accepted.
    let checked = |name: &str, f: fn(&RunSample) -> f64| -> anyhow::Result<Spread> {
        let in_order: Vec<f64> = samples.iter().map(f).collect();
        let spread = Spread::of(in_order.clone()).ok_or_else(missing)?;
        assert_no_drift(name, &in_order, spread.median)?;
        Ok(spread)
    };

    let ttft_ms = checked("ttft_ms", |s| s.ttft_ms)?;
    let decode_tps = checked("decode_tps", |s| s.decode_tps)?;
    let itl_p50_ms = checked("itl_p50_ms", |s| s.itl_p50_ms)?;
    #[allow(
        clippy::cast_precision_loss,
        reason = "KV byte totals are far below 2^53; f64 is exact over the whole range"
    )]
    let kv_bytes = checked("kv_cache_bytes", |s| s.kv_cache_bytes as f64)?;

    // `itl_p99_ms` is deliberately NOT drift-checked. Nearest-rank p99 over a
    // 128-token run is the second-largest inter-token gap: an extreme-order
    // statistic whose run-to-run movement is dominated by whether that one run
    // happened to hit a stall, not by whether the cell has settled. Measured
    // gemma-4-e2b at 4k: ttft, decode TPS and ITL p50 all moved 5-6% across
    // three runs while p99 moved 26% — checking it would abort cells whose
    // measurements are fine. Its spread is still printed, so the tail stays
    // visible; it is just not evidence of a trend.
    let itl_p99_ms =
        Spread::of(samples.iter().map(|s| s.itl_p99_ms).collect()).ok_or_else(missing)?;

    // Prefill throughput is present only if every run produced one.
    let prefill_in_order: Option<Vec<f64>> = samples.iter().map(|s| s.prefill_tps).collect();
    let prefill_tps = match prefill_in_order {
        Some(v) => {
            let spread = Spread::of(v.clone()).ok_or_else(missing)?;
            assert_no_drift("prefill_tps", &v, spread.median)?;
            Some(spread)
        }
        None => None,
    };

    let token_digest = samples.first().ok_or_else(missing)?.token_digest;

    Ok(BenchSummary {
        ttft_ms,
        decode_tps,
        prefill_tps,
        itl_p50_ms,
        itl_p99_ms,
        kv_bytes,
        token_digest,
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

/// Check that a run decoded the same tokens as the cell's first run.
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
fn assert_same_token_stream(label: &str, expected: u64, got: u64) -> anyhow::Result<()> {
    if expected == got {
        return Ok(());
    }
    Err(anyhow::anyhow!(
        "{label} decoded a different token stream than the first run of this cell \
         (digest {got:#018x} vs {expected:#018x}). Every run here decodes the same prompt at \
         temperature 0, so the runs are not repeats of one measurement — and a cache that \
         stops being written decodes faster while producing wrong tokens. Refusing to \
         summarise them as one cell"
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
    // the same defect and costs nothing to catch.
    let mut expected_digest: Option<u64> = None;
    let mut check_digest = |label: &str, s: &RunSample| -> anyhow::Result<()> {
        match expected_digest {
            None => {
                expected_digest = Some(s.token_digest);
                Ok(())
            }
            Some(e) => assert_same_token_stream(label, e, s.token_digest),
        }
    };

    for i in 0..args.warmup {
        info!(run = i + 1, of = args.warmup, "bench: warmup run");
        let s = measure_one(model, tokenizer, prompt_ids, args)
            .map_err(|e| anyhow::anyhow!("warmup run {} failed: {e}", i + 1))?;
        check_digest(&format!("warmup run {}", i + 1), &s)?;
    }

    let mut samples: Vec<RunSample> = Vec::with_capacity(args.runs as usize);
    for i in 0..args.runs {
        let s = measure_one(model, tokenizer, prompt_ids, args)
            .map_err(|e| anyhow::anyhow!("measured run {} failed: {e}", i + 1))?;
        check_digest(&format!("measured run {}", i + 1), &s)?;
        info!(
            run = i + 1,
            of = args.runs,
            ttft_ms = s.ttft_ms,
            decode_tps = s.decode_tps,
            itl_p50_ms = s.itl_p50_ms,
            itl_p99_ms = s.itl_p99_ms,
            kv_cache_bytes = s.kv_cache_bytes,
            token_digest = format!("{:#018x}", s.token_digest),
            "bench: measured run"
        );
        samples.push(s);
    }
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
