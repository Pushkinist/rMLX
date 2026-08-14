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
//!    2 so there is always a spread to report.
//!
//! `bench` is read-only with respect to the metrics database: it prints, it
//! does not record. Use `rmlx baseline --record` for the append-only store.

#![allow(clippy::needless_pass_by_value, clippy::too_many_lines)]

use std::path::{Path, PathBuf};
use std::time::Instant;

use rmlx_mlx::Device;
use rmlx_models::arch;
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
/// Zero, deliberately. `bench` runs N generations of the *same* prompt in one
/// process; with any slots at all, run 2 onwards would be served from the
/// prompt cache, skip prefill entirely, and report a TTFT of a few milliseconds
/// that looks like a very fast prefill instead of a measurement that was never
/// taken. Zero slots makes every run a cache miss and therefore a real prefill.
/// `assert_prefill_measured` re-checks that per run rather than trusting this
/// constant.
const BENCH_PROMPT_CACHE_SLOTS: usize = 0;

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
// KV-byte reading — three states, not two
// ---------------------------------------------------------------------------

/// What a pair of `kv_cache_bytes_sample()` reads around one generation says
/// about the byte count that generation produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvBytesVerdict {
    /// The generation reported a non-zero byte count. Usable.
    Reported(u64),
    /// The generation reported nothing: the readable value belongs to an
    /// earlier generation, or is the never-written initialiser. This is the
    /// state that a bare `-> u64` accessor collapses into "0" or, worse, into
    /// the previous run's plausible-looking figure.
    Unreported,
    /// The generation did report, and reported zero bytes. Distinct from
    /// `Unreported`: the plumbing works and the answer is still not usable as a
    /// resident-KV measurement after a real prefill.
    ReportedZero,
}

/// Classify the byte count a generation produced from the store sequence
/// observed before and after it.
///
/// Detection ("did this generation report?") is decided by the sequence, and
/// only then is the value interpreted. Collapsing the two — treating a zero, or
/// an unchanged sequence, as "no KV" — is what silently records one run's
/// number under another run's label.
pub(crate) const fn classify_kv_bytes(
    before: rmlx_models::KvBytesSample,
    after: rmlx_models::KvBytesSample,
) -> KvBytesVerdict {
    if after.seq <= before.seq {
        return KvBytesVerdict::Unreported;
    }
    if after.bytes == 0 {
        return KvBytesVerdict::ReportedZero;
    }
    KvBytesVerdict::Reported(after.bytes)
}

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
/// `None` when the arch has no cache to report. Two conditions, both necessary:
///
/// - **`hits == 0`.** A hit means the post-prefill KV snapshot was replayed, so
///   the run's TTFT is a cache-replay time, not a time-to-first-token — a
///   number that is small, stable, and wrong.
/// - **`misses >= 1`.** The cache was consulted and did not serve this run, so
///   a prefill genuinely happened. Counters that report nothing certify
///   nothing.
///
/// Absolute counters rather than a before/after delta, deliberately: a cache
/// configured with zero slots is rebuilt on each generation, which resets the
/// counters, and a delta across a reset reads as "no activity". The rule holds
/// either way — under stable counters a repeat that was served shows `hits > 0`,
/// and under reset counters the freshly-built cache shows `hits == 0,
/// misses == 1`.
pub(crate) fn assert_prefill_measured(stats: Option<(u64, u64)>) -> anyhow::Result<()> {
    let Some((hits, misses)) = stats else {
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

/// Everything one generation contributes to the summary.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RunSample {
    /// Prefill through first token, milliseconds.
    pub ttft_ms: f64,
    /// Steady-state decode rate over tokens 2..N, tokens/second.
    pub decode_tps: f64,
    /// Prefill throughput, prompt tokens/second.
    pub prefill_tps: f64,
    /// Median inter-token latency within the run, milliseconds.
    pub itl_p50_ms: f64,
    /// 99th-percentile inter-token latency within the run, milliseconds.
    pub itl_p99_ms: f64,
    /// Filled-prefix KV cache bytes after decode.
    pub kv_cache_bytes: u64,
    /// Tokens actually generated.
    pub n_generated: usize,
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
    let prefill_tps = if first > 0.0 && prompt_tokens > 0 {
        prompt_tokens as f64 / first
    } else {
        0.0
    };

    Some(RunSample {
        ttft_ms: first * 1000.0,
        decode_tps,
        prefill_tps,
        itl_p50_ms: percentile_sorted(&itl_ms, 0.50)?,
        itl_p99_ms: percentile_sorted(&itl_ms, 0.99)?,
        kv_cache_bytes,
        n_generated: arrivals_s.len(),
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

    assert_prefill_measured(model.cache_stats().map(|s| (s.hits, s.misses)))?;

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

    sample_from_arrivals(&arrivals_s, prompt_ids.len(), kv_cache_bytes).ok_or_else(|| {
        anyhow::anyhow!(
            "run generated {} token(s); at least 2 are needed for an inter-token latency \
             and a steady-state decode rate. Raise --max-tokens",
            arrivals_s.len()
        )
    })
}

/// Summary of one bench cell.
struct BenchSummary {
    ttft_ms: Spread,
    decode_tps: Spread,
    prefill_tps: Spread,
    itl_p50_ms: Spread,
    itl_p99_ms: Spread,
    kv_bytes: Spread,
}

/// Fold the measured runs into per-metric spreads.
///
/// `None` when `samples` is empty; every metric is summarised from the same
/// runs, so either all of them exist or none do.
fn summarize(samples: &[RunSample]) -> Option<BenchSummary> {
    let pick = |f: fn(&RunSample) -> f64| Spread::of(&samples.iter().map(f).collect::<Vec<_>>());
    Some(BenchSummary {
        ttft_ms: pick(|s| s.ttft_ms)?,
        decode_tps: pick(|s| s.decode_tps)?,
        prefill_tps: pick(|s| s.prefill_tps)?,
        itl_p50_ms: pick(|s| s.itl_p50_ms)?,
        itl_p99_ms: pick(|s| s.itl_p99_ms)?,
        #[allow(
            clippy::cast_precision_loss,
            reason = "KV byte totals are far below 2^53; f64 is exact over the whole range"
        )]
        kv_bytes: pick(|s| s.kv_cache_bytes as f64)?,
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
    for i in 0..args.warmup {
        info!(run = i + 1, of = args.warmup, "bench: warmup run");
        measure_one(model, tokenizer, prompt_ids, args)
            .map_err(|e| anyhow::anyhow!("warmup run {} failed: {e}", i + 1))?;
    }

    let mut samples: Vec<RunSample> = Vec::with_capacity(args.runs as usize);
    for i in 0..args.runs {
        let s = measure_one(model, tokenizer, prompt_ids, args)
            .map_err(|e| anyhow::anyhow!("measured run {} failed: {e}", i + 1))?;
        info!(
            run = i + 1,
            of = args.runs,
            ttft_ms = s.ttft_ms,
            decode_tps = s.decode_tps,
            itl_p50_ms = s.itl_p50_ms,
            itl_p99_ms = s.itl_p99_ms,
            kv_cache_bytes = s.kv_cache_bytes,
            "bench: measured run"
        );
        samples.push(s);
    }
    Ok(samples)
}

/// Execute `rmlx bench`.
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
    let summary = summarize(&samples)
        .ok_or_else(|| anyhow::anyhow!("no measured runs completed — nothing to summarise"))?;

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
        "metrics": {
            "ttft_ms": spread_json(&s.ttft_ms),
            "decode_tps": spread_json(&s.decode_tps),
            "prefill_tps": spread_json(&s.prefill_tps),
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
    row("prefill_tps", &s.prefill_tps, 1);
    row("kv_cache_bytes", &s.kv_bytes, 0);

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
