//! Unit tests for the `rmlx bench` instrument.
//!
//! The subject here is the instrument's refusal behaviour, not its arithmetic:
//! every test that matters feeds it a condition under which its numbers would
//! be wrong and asserts that it says so instead of returning one.

use super::*;
use rmlx_models::KvBytesSample;

fn sample(bytes: u64, seq: u64) -> KvBytesSample {
    // `KvBytesSample` is `#[non_exhaustive]` outside its crate, so build it
    // from the `Default` and assign — the same way any downstream consumer
    // would have to.
    let mut s = KvBytesSample::default();
    s.bytes = bytes;
    s.seq = seq;
    s
}

/// Prompt-cache counters as the guard sees them. Named fields, so a
/// hits/misses transposition cannot slip through the way it could with a
/// positional tuple.
fn cache_stats(hits: u64, misses: u64) -> CacheStats {
    // `CacheStats` is `#[non_exhaustive]` outside its crate, so build it from
    // the `Default` and assign — the same way any downstream consumer would.
    let mut s = CacheStats::default();
    s.hits = hits;
    s.misses = misses;
    s
}

// ── KV bytes: three states, not two ─────────────────────────────────

/// The load-bearing case. A generation that never reached its KV-byte store
/// leaves the previous generation's figure readable. A classifier that looked
/// only at the value would hand back `9_000_000` — a number from a different
/// run, indistinguishable from a fresh one.
#[test]
fn kv_bytes_unreported_when_sequence_did_not_advance() {
    let before = sample(9_000_000, 7);
    let after = sample(9_000_000, 7);
    assert_eq!(
        classify_kv_bytes(before, after),
        KvBytesVerdict::Unreported,
        "an unchanged store sequence means this generation reported nothing; the \
         readable byte count belongs to an earlier one"
    );
}

/// "Never stored" is the same failure with a zero value: the byte count is the
/// initialiser, not a measurement of an empty cache.
#[test]
fn kv_bytes_unreported_when_nothing_was_ever_stored() {
    assert_eq!(
        classify_kv_bytes(sample(0, 0), sample(0, 0)),
        KvBytesVerdict::Unreported
    );
}

/// A genuine store of zero is a *different* state from "nothing was stored" —
/// the plumbing worked and the answer is zero. It is still not usable, but it
/// points at the byte accounting rather than at the reporting path.
#[test]
fn kv_bytes_reported_zero_is_distinct_from_unreported() {
    assert_eq!(
        classify_kv_bytes(sample(0, 0), sample(0, 1)),
        KvBytesVerdict::ReportedZero
    );
    assert_ne!(
        classify_kv_bytes(sample(0, 0), sample(0, 1)),
        classify_kv_bytes(sample(0, 0), sample(0, 0)),
        "collapsing 'reported zero' into 'never reported' is the bug this \
         classifier exists to prevent"
    );
}

#[test]
fn kv_bytes_reported_when_sequence_advanced_with_a_value() {
    assert_eq!(
        classify_kv_bytes(sample(9_000_000, 7), sample(4_242, 8)),
        KvBytesVerdict::Reported(4_242)
    );
}

/// Both unusable verdicts must produce an error, and the two errors must not
/// read the same — the operator needs to know which of the two happened.
#[test]
fn unusable_kv_verdicts_are_errors_with_distinct_reasons() {
    let unreported = kv_bytes_or_reason(KvBytesVerdict::Unreported, "TestArch");
    let zero = kv_bytes_or_reason(KvBytesVerdict::ReportedZero, "TestArch");
    let unreported_msg = unreported
        .expect_err("Unreported must not yield a number")
        .to_string();
    let zero_msg = zero
        .expect_err("ReportedZero must not yield a number")
        .to_string();
    assert!(
        unreported_msg.contains("store sequence did not advance"),
        "unreported error must name the cause, got: {unreported_msg}"
    );
    assert!(
        zero_msg.contains("0 bytes"),
        "reported-zero error must name the cause, got: {zero_msg}"
    );
    assert_ne!(unreported_msg, zero_msg);
    assert_eq!(
        kv_bytes_or_reason(KvBytesVerdict::Reported(512), "TestArch").ok(),
        Some(512)
    );
}

// ── Prompt-cache guard: a TTFT is only a TTFT when prefill ran ──────

/// The condition an in-process repeated-run bench walks straight into: with
/// reusing prompt-cache slots, run 2 replays the post-prefill snapshot and its
/// "TTFT" is a few milliseconds of cache replay.
#[test]
fn prompt_cache_hit_is_refused() {
    let err = assert_prefill_measured(Some(&cache_stats(1, 1)))
        .expect_err("a cache-served run must not pass as a prefill measurement");
    let msg = err.to_string();
    assert!(msg.contains("served from the prompt cache"), "got: {msg}");
}

/// A cache that was never consulted certifies nothing either — silence is not
/// evidence that a fresh prefill happened.
#[test]
fn no_prompt_cache_activity_is_refused() {
    let err = assert_prefill_measured(Some(&cache_stats(0, 0)))
        .expect_err("zero activity certifies nothing");
    assert!(err.to_string().contains("never consulted"));
}

/// An arch with no cache stats at all is the same failure: nothing observable
/// says a prefill happened.
#[test]
fn absent_prompt_cache_stats_are_refused() {
    let err = assert_prefill_measured(None).expect_err("absent stats certify nothing");
    assert!(err.to_string().contains("no prompt-cache stats"));
}

/// Repeated misses are fine — under counters that are not reset between runs,
/// run 3 legitimately shows 3 cumulative misses. What matters is that none of
/// them was a hit.
#[test]
fn repeated_misses_with_no_hits_pass() {
    assert!(assert_prefill_measured(Some(&cache_stats(0, 1))).is_ok());
    assert!(assert_prefill_measured(Some(&cache_stats(0, 3))).is_ok());
}

/// A hit is refused however many misses accompany it — a run that was served
/// is a run whose prefill did not happen.
#[test]
fn a_hit_is_refused_regardless_of_miss_count() {
    assert!(assert_prefill_measured(Some(&cache_stats(1, 5))).is_err());
    assert!(assert_prefill_measured(Some(&cache_stats(7, 0))).is_err());
}

// ── Spread: no central value without its range ─────────────────────

#[test]
fn spread_of_empty_sample_is_none() {
    assert_eq!(
        Spread::of(Vec::new()),
        None,
        "a median of nothing is not zero"
    );
}

#[test]
fn spread_reports_median_and_observed_range() {
    let s = Spread::of(vec![110.0, 100.0, 105.0]).expect("non-empty");
    assert!((s.median - 105.0).abs() < 1e-9);
    assert!((s.min - 100.0).abs() < 1e-9);
    assert!((s.max - 110.0).abs() < 1e-9);
    assert_eq!(s.n, 3);
    // 10/105 → ~9.52%
    assert!(
        (s.range_pct() - 9.523_809_5).abs() < 1e-4,
        "{}",
        s.range_pct()
    );
}

#[test]
fn spread_median_of_even_sample_averages_the_middle_two() {
    let s = Spread::of(vec![1.0, 2.0, 3.0, 4.0]).expect("non-empty");
    assert!((s.median - 2.5).abs() < 1e-9);
}

/// Two samples with the same median but different stability must not print the
/// same thing — that collapse is what made single-run numbers look conclusive.
#[test]
fn equal_medians_with_different_stability_are_distinguishable() {
    let tight = Spread::of(vec![99.9, 100.0, 100.1]).expect("non-empty");
    let loose = Spread::of(vec![70.0, 100.0, 130.0]).expect("non-empty");
    assert!((tight.median - loose.median).abs() < 1e-9);
    assert!(
        loose.range_pct() > tight.range_pct() * 10.0,
        "tight={} loose={}",
        tight.range_pct(),
        loose.range_pct()
    );
}

/// The instrument has no single-run mode. The check fires before any file is
/// read, so a bogus model path cannot be what produced the error.
#[test]
fn single_run_invocation_is_refused_before_any_io() {
    let args = BenchArgs {
        model: PathBuf::from("/nonexistent/model"),
        prompt: PathBuf::from("/nonexistent/prompt.txt"),
        prompt_label: "x".to_owned(),
        device: Device::Cpu,
        max_tokens: 8,
        runs: 1,
        warmup: 0,
        kv_quant: rmlx_kv_quant::KvQuant::None,
        max_ctx: None,
        max_prompt_tokens: 4096,
        cap_is_explicit: false,
        allow_truncate: false,
        json: false,
    };
    let msg = run_bench(args)
        .expect_err("--runs 1 must be refused")
        .to_string();
    assert!(
        msg.contains("--runs must be at least"),
        "must refuse for lack of spread, not for the missing paths: {msg}"
    );
}

// ── Percentiles ─────────────────────────────────────────────────────

#[test]
fn percentile_of_empty_sample_is_none() {
    assert_eq!(percentile_sorted(&[], 0.5), None);
}

#[test]
fn percentile_nearest_rank() {
    let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    assert_eq!(percentile_sorted(&v, 0.5), Some(5.0));
    assert_eq!(percentile_sorted(&v, 0.99), Some(10.0));
    assert_eq!(percentile_sorted(&v, 0.0), Some(1.0));
    assert_eq!(percentile_sorted(&v, 1.0), Some(10.0));
}

/// p99 must track the tail, not the bulk. A 2%-of-samples stall is exactly what
/// a mean hides and what the instrument exists to surface.
///
/// Nearest-rank p99 of 100 samples is the 99th smallest, so it reports the
/// stall once the tail is at least 2 samples wide — a single 1-in-100 outlier
/// legitimately sits above p99 and is p100's business.
#[test]
fn p99_surfaces_a_tail_a_mean_would_hide() {
    let mut v: Vec<f64> = std::iter::repeat_n(10.0, 98).collect();
    v.push(200.0);
    v.push(200.0);
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    assert_eq!(percentile_sorted(&v, 0.50), Some(10.0));
    assert_eq!(percentile_sorted(&v, 0.99), Some(200.0));
    assert!(
        mean < 15.0,
        "the mean stays near the bulk ({mean}) — only p99 shows the stall"
    );
}

#[test]
fn percentile_rejects_out_of_range_quantile() {
    assert_eq!(percentile_sorted(&[1.0], 1.5), None);
    assert_eq!(percentile_sorted(&[1.0], -0.1), None);
}

// ── Per-run derivation from token arrival times ─────────────────────

#[test]
fn sample_from_arrivals_derives_ttft_itl_and_decode_rate() {
    // TTFT 0.5 s, then 4 more tokens at 100 ms each.
    let arrivals = [0.5, 0.6, 0.7, 0.8, 0.9];
    let s = sample_from_arrivals(&arrivals, 1000, 4096, 0xabc).expect("5 tokens is enough");
    assert!((s.ttft_ms - 500.0).abs() < 1e-6);
    assert!((s.itl_p50_ms - 100.0).abs() < 1e-6);
    assert!((s.itl_p99_ms - 100.0).abs() < 1e-6);
    // 4 inter-token intervals over 0.4 s.
    assert!((s.decode_tps - 10.0).abs() < 1e-6, "{}", s.decode_tps);
    // 1000 prompt tokens in 0.5 s.
    assert!((s.prefill_tps.expect("first arrival is non-zero") - 2000.0).abs() < 1e-6);
    assert_eq!(s.kv_cache_bytes, 4096);
    assert_eq!(s.n_generated, 5);
}

/// The decode rate must exclude prefill. If TTFT leaked into the denominator
/// this run would report ~5.5 tok/s instead of 10.
#[test]
fn decode_rate_excludes_prefill() {
    let s =
        sample_from_arrivals(&[0.5, 0.6, 0.7, 0.8, 0.9], 1000, 1, 0xabc).expect("enough tokens");
    let with_prefill_included = 5.0 / 0.9;
    assert!(
        s.decode_tps > with_prefill_included * 1.5,
        "decode_tps {} looks like it still carries the prefill cost ({with_prefill_included})",
        s.decode_tps
    );
}

/// One token gives no inter-token interval. Falling back to the combined
/// prefill+decode rate here would print a prefill-dominated figure under a
/// decode-rate label.
#[test]
fn single_token_run_yields_no_sample() {
    assert!(sample_from_arrivals(&[0.5], 100, 4096, 0).is_none());
    assert!(sample_from_arrivals(&[], 100, 4096, 0).is_none());
}

/// Arrivals with no elapsed time between first and last cannot produce a rate;
/// a divide-by-zero here would surface as `inf` tok/s.
#[test]
fn zero_width_decode_window_yields_no_sample() {
    assert!(sample_from_arrivals(&[0.5, 0.5], 100, 4096, 0).is_none());
}

/// ITL percentiles are computed over the gaps, so a stall in the middle of a
/// run shows up in p99 while p50 stays at the healthy cadence.
#[test]
fn itl_p99_catches_a_mid_run_stall() {
    let mut arrivals = vec![0.1];
    for i in 1..100 {
        // 10 ms cadence, with one 500 ms stall at step 50.
        let step = if i == 50 { 0.5 } else { 0.01 };
        let prev = *arrivals.last().expect("non-empty");
        arrivals.push(prev + step);
    }
    let s = sample_from_arrivals(&arrivals, 100, 4096, 0xabc).expect("enough tokens");
    assert!((s.itl_p50_ms - 10.0).abs() < 1e-6, "p50={}", s.itl_p50_ms);
    assert!(
        s.itl_p99_ms > 400.0,
        "p99={} must expose the stall",
        s.itl_p99_ms
    );
}

// ── Load guard ──────────────────────────────────────────────────────

#[test]
fn load_guard_flags_a_busy_host() {
    assert!(load_is_contended(16.0, 16));
    assert!(load_is_contended(20.0, 16));
    assert!(!load_is_contended(1.5, 16));
}

#[test]
fn loadavg_parses_the_sysctl_shape() {
    assert_eq!(parse_loadavg_1m("{ 3.03 4.53 5.24 }\n"), Some(3.03));
    assert_eq!(parse_loadavg_1m("{ 0.00 0.00 0.00 }"), Some(0.0));
}

/// An unreadable load must read as "no reading", not as an idle host — the
/// latter would stamp a contended measurement as quiet-machine.
#[test]
fn unparseable_loadavg_is_absent_not_zero() {
    assert_eq!(parse_loadavg_1m(""), None);
    assert_eq!(parse_loadavg_1m("{ }"), None);
    assert_eq!(parse_loadavg_1m("sysctl: unknown oid"), None);
}

// ── Drift: a trend is not a spread ──────────────────────────────────

/// The measured case this guard was built for: gemma-4-e2b at 64k, three
/// consecutive in-process generations, TTFT in milliseconds. `Spread` sorts,
/// so it reports this as a wide range around a median; in collection order it
/// is a ramp, and the median is a point on it.
#[test]
fn measured_ttft_ramp_is_detected_as_drift() {
    let in_order = [17_769.4, 22_764.3, 25_475.1];
    let spread = Spread::of(in_order.to_vec()).expect("non-empty");
    let pct = detect_drift(&in_order, spread.median).expect("a 43% ramp is a trend");
    assert!(
        pct > 30.0,
        "fitted drift {pct:.1}% understates a 17.8s → 25.5s ramp"
    );
    let err = assert_no_drift("ttft_ms", &in_order, spread.median)
        .expect_err("a drifting metric must not be summarised");
    let msg = err.to_string();
    assert!(msg.contains("trend, not a spread"), "got: {msg}");
    assert!(msg.contains("rose"), "must name the direction: {msg}");
}

/// Order is the whole point: a one-run spike that starts and ends at the same
/// place is spread, while the same magnitude of movement spent going one way is
/// a trend. `Spread` cannot tell them apart, because it sorts before it looks —
/// both of these produce an identical median and an identical range.
#[test]
fn a_spike_is_spread_but_a_ramp_is_drift() {
    let spike = [100.0, 130.0, 100.0];
    let ramp = [100.0, 115.0, 130.0];
    let spike_spread = Spread::of(spike.to_vec()).expect("non-empty");
    let ramp_spread = Spread::of(ramp.to_vec()).expect("non-empty");
    assert!(
        (spike_spread.max - ramp_spread.max).abs() < 1e-9
            && (spike_spread.min - ramp_spread.min).abs() < 1e-9,
        "both samples must look identical to the order-blind summary"
    );
    assert_eq!(
        detect_drift(&spike, spike_spread.median),
        None,
        "a run that returned to where it started did not trend"
    );
    assert!(
        detect_drift(&ramp, ramp_spread.median).is_some(),
        "the same movement spent going one way is a trend"
    );
}

/// Decode TPS falling across runs is the same defect with the opposite sign —
/// the measured Bonsai case, 129.24 → 126.74 → 115.59.
#[test]
fn a_falling_metric_drifts_too() {
    let in_order = [129.24, 126.74, 115.59];
    let spread = Spread::of(in_order.to_vec()).expect("non-empty");
    let pct = detect_drift(&in_order, spread.median).expect("a 10.6% decline is a trend");
    assert!(pct < 0.0, "direction must be signed");
    let msg = assert_no_drift("decode_tps", &in_order, spread.median)
        .expect_err("a falling metric is still drifting")
        .to_string();
    assert!(msg.contains("fell"), "got: {msg}");
}

/// A settled cell must pass. These are the measured gemma-4-e2b ITL p50 values
/// from the same run that produced the TTFT ramp above: the run-to-run noise is
/// under 1%, and refusing it would make the instrument useless.
#[test]
fn a_settled_metric_is_not_drift() {
    let in_order = [9.455, 9.840, 9.538];
    let spread = Spread::of(in_order.to_vec()).expect("non-empty");
    assert_eq!(detect_drift(&in_order, spread.median), None);
    assert!(assert_no_drift("itl_p50_ms", &in_order, spread.median).is_ok());
}

/// A constant metric has no trend and no scale problem. Measured KV bytes are
/// bit-identical across runs of a cell, so this is the common case.
#[test]
fn a_constant_metric_is_not_drift() {
    let in_order = [649_353_216.0, 649_353_216.0, 649_353_216.0];
    assert_eq!(detect_drift(&in_order, 649_353_216.0), None);
}

/// Measured gemma-4-e2b at 4k, three runs: the robust metrics move 5-6% and the
/// tail statistic moves 26%. The threshold has to sit above the first group and
/// below a real ramp, or the guard is either useless or unusable. This pins
/// that separation, so a later threshold change has to face it.
#[test]
fn the_threshold_separates_warmup_wobble_from_a_ramp() {
    // Settled-enough: measured ttft, decode TPS and ITL p50 from one cell.
    for (name, v, med) in [
        ("ttft_ms", [209.207, 215.761, 198.285], 209.207),
        ("decode_tps", [119.178, 121.846, 126.562], 121.846),
        ("itl_p50_ms", [8.368, 8.098, 7.889], 8.098),
    ] {
        assert_eq!(
            detect_drift(&v, med),
            None,
            "{name} is ordinary warmup wobble and must not abort the cell"
        );
    }
    // A real ramp from the same model at 64k must still be caught.
    assert!(detect_drift(&[17_065.0, 22_768.2, 24_684.6], 22_768.2).is_some());
}

/// Mutation check: the guard must key on the run-to-run *trend*, not on the
/// magnitude of the values. A cell an order of magnitude larger, equally
/// settled, must still pass.
#[test]
fn drift_is_scale_free() {
    let small = [100.0, 101.0, 100.5];
    let large = [100_000.0, 101_000.0, 100_500.0];
    assert_eq!(detect_drift(&small, 100.5), None);
    assert_eq!(detect_drift(&large, 100_500.0), None);
    // And a proportionally identical ramp is caught at either scale.
    assert!(detect_drift(&[100.0, 120.0, 140.0], 120.0).is_some());
    assert!(detect_drift(&[100_000.0, 120_000.0, 140_000.0], 120_000.0).is_some());
}

/// Degenerate inputs must not manufacture a verdict: one run has no order, and
/// a zero median has no scale to express a change against.
#[test]
fn drift_needs_order_and_a_scale() {
    assert_eq!(detect_drift(&[100.0], 100.0), None);
    assert_eq!(detect_drift(&[], 0.0), None);
    assert_eq!(detect_drift(&[0.0, 5.0], 0.0), None);
}

// ── Token stream: same prompt, temperature 0, same tokens ───────────

/// A frozen KV cache decodes *faster* while producing wrong tokens, so a
/// timing-only instrument is biased toward accepting it. The digest is what
/// makes that visible.
#[test]
fn a_differing_token_stream_is_refused() {
    let a = token_stream_digest([1_u32, 2, 3, 4]);
    let b = token_stream_digest([1_u32, 2, 3, 5]);
    assert_ne!(a, b, "one differing token must change the digest");
    let msg = assert_same_token_stream("measured run 2", a, b)
        .expect_err("runs of one cell must decode the same tokens")
        .to_string();
    assert!(msg.contains("different token stream"), "got: {msg}");
    assert!(assert_same_token_stream("measured run 2", a, a).is_ok());
}

/// Order matters: a digest that ignored it would pass a run that emitted the
/// right tokens in the wrong sequence.
#[test]
fn the_digest_is_order_sensitive() {
    assert_ne!(
        token_stream_digest([1_u32, 2, 3]),
        token_stream_digest([3_u32, 2, 1])
    );
}

/// Length matters too — a truncated stream must not collide with a full one.
#[test]
fn the_digest_distinguishes_a_truncated_stream() {
    assert_ne!(
        token_stream_digest([1_u32, 2, 3]),
        token_stream_digest([1_u32, 2])
    );
    assert_ne!(token_stream_digest([0_u32]), token_stream_digest([]));
}

// ── prefill_tps: absent, never zero ─────────────────────────────────

/// The one place in the file that used to substitute a literal `0.0` for a
/// quantity it could not compute, and then feed it to `Spread` and print it as
/// a measurement.
#[test]
fn prefill_throughput_is_absent_not_zero_when_undefined() {
    // Prompt of zero tokens: nothing was prefilled, so there is no throughput.
    let s = sample_from_arrivals(&[0.5, 0.6, 0.7], 0, 4096, 0xabc).expect("enough tokens");
    assert_eq!(
        s.prefill_tps, None,
        "a zero here is printed exactly like a measured throughput of zero"
    );
    // The rest of the run is still a valid measurement.
    assert!((s.ttft_ms - 500.0).abs() < 1e-6);
}

/// A run whose first token arrived at t=0 has no prefill interval to divide by.
#[test]
fn prefill_throughput_is_absent_when_the_first_token_is_instant() {
    let s = sample_from_arrivals(&[0.0, 0.1, 0.2], 1000, 4096, 0xabc).expect("enough tokens");
    assert_eq!(s.prefill_tps, None);
}
