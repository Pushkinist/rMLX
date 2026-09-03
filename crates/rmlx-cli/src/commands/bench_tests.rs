// LOC-exempt: sibling test file for one command. Its length tracks the number of
// refusals `rmlx bench` makes, and each refusal's test carries the reasoning for
// why that measurement condition is load-bearing — prose that is the point, not
// padding. Splitting by topic would separate a guard from the argument for it.
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

/// The same counters, plus the SSD-tier hydrate count.
fn cache_stats_ssd(hits: u64, misses: u64, ssd_hits: u64) -> CacheStats {
    let mut s = cache_stats(hits, misses);
    s.ssd_hits = ssd_hits;
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

/// The shape that gets past a hits/misses-only guard: emptying the RAM slots
/// does not detach the SSD source, so the run misses in RAM (`hits == 0`,
/// `misses == 1` — a clean bill of health) and is then served by hydrating a
/// `.kvb`. Its "TTFT" is a reconstruction time.
#[test]
fn an_ssd_hydrate_is_refused_even_though_it_looks_like_a_clean_miss() {
    let hydrated = cache_stats_ssd(0, 1, 1);
    assert!(
        assert_prefill_measured(Some(&cache_stats(0, 1))).is_ok(),
        "the same counters without the hydrate are a genuine prefill"
    );
    let msg = assert_prefill_measured(Some(&hydrated))
        .expect_err("a hydrated run did not prefill")
        .to_string();
    assert!(msg.contains("SSD KV tier"), "got: {msg}");
    // And it is a distinct diagnosis from a RAM hit — the operator needs to
    // know which tier served it.
    let ram_hit = assert_prefill_measured(Some(&cache_stats(1, 1)))
        .expect_err("a RAM hit did not prefill either")
        .to_string();
    assert_ne!(msg, ram_hit);
}

// ── Spread: no central value without its range ─────────────────────

#[test]
fn spread_of_empty_sample_is_none() {
    assert_eq!(Spread::of(&[]), None, "a median of nothing is not zero");
}

#[test]
fn spread_reports_median_and_observed_range() {
    let s = Spread::of(&[110.0, 100.0, 105.0]).expect("non-empty");
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
    let s = Spread::of(&[1.0, 2.0, 3.0, 4.0]).expect("non-empty");
    assert!((s.median - 2.5).abs() < 1e-9);
}

/// Two samples with the same median but different stability must not print the
/// same thing — that collapse is what made single-run numbers look conclusive.
#[test]
fn equal_medians_with_different_stability_are_distinguishable() {
    let tight = Spread::of(&[99.9, 100.0, 100.1]).expect("non-empty");
    let loose = Spread::of(&[70.0, 100.0, 130.0]).expect("non-empty");
    assert!((tight.median - loose.median).abs() < 1e-9);
    assert!(
        loose.range_pct() > tight.range_pct() * 10.0,
        "tight={} loose={}",
        tight.range_pct(),
        loose.range_pct()
    );
}

/// A greedy, otherwise-valid argument bundle pointing at paths that do not
/// exist, so any error a test observes came from an argument check rather than
/// from I/O.
fn unrunnable_args(runs: u32) -> BenchArgs {
    BenchArgs {
        model: PathBuf::from("/nonexistent/model"),
        prompt: PathBuf::from("/nonexistent/prompt.txt"),
        prompt_label: "x".to_owned(),
        device: Device::Cpu,
        max_tokens: 8,
        runs,
        warmup: 0,
        kv_quant: rmlx_kv_quant::KvQuant::None,
        max_ctx: None,
        max_prompt_tokens: Some(4096),
        allow_truncate: false,
        json: false,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
    }
}

/// The instrument has no single-run mode. The check fires before any file is
/// read, so a bogus model path cannot be what produced the error.
#[test]
fn single_run_invocation_is_refused_before_any_io() {
    let msg = run_bench(unrunnable_args(1))
        .expect_err("--runs 1 must be refused")
        .to_string();
    assert!(
        msg.contains("--runs must be at least"),
        "must refuse for lack of spread, not for the missing paths: {msg}"
    );
}

// ── Sampler knobs ───────────────────────────────────────────────────

/// A NaN temperature fails the decode loop's `> 0.0` gate silently: the cell
/// would run greedy while every label on it said otherwise. Rejected up front,
/// before the model is loaded.
#[test]
fn nan_temperature_is_refused() {
    let mut args = unrunnable_args(3);
    args.temperature = f32::NAN;
    let msg = run_bench(args)
        .expect_err("NaN temperature must be refused")
        .to_string();
    assert!(msg.contains("--temperature"), "wrong rejection: {msg}");
}

#[test]
fn negative_temperature_is_refused() {
    let mut args = unrunnable_args(3);
    args.temperature = -0.5;
    let msg = run_bench(args)
        .expect_err("negative temperature must be refused")
        .to_string();
    assert!(msg.contains("--temperature"), "wrong rejection: {msg}");
}

/// `top_p` outside `(0, 1]` filters every token or none of them; neither is a
/// nucleus. The sampler's own gate would quietly no-op instead.
#[test]
fn out_of_range_top_p_is_refused() {
    for bad in [0.0f32, -0.1, 1.5, f32::NAN] {
        let mut args = unrunnable_args(3);
        args.temperature = 0.7;
        args.top_p = bad;
        let msg = run_bench(args)
            .expect_err("out-of-range --top-p must be refused")
            .to_string();
        assert!(msg.contains("--top-p"), "wrong rejection for {bad}: {msg}");
    }
}

/// The binary bounds temperature to [0, 2] on the HTTP surface and on
/// `--default-temperature`. Above that the distribution flattens toward uniform
/// over the vocabulary; at infinity it is exactly uniform. Benching a cell no
/// served request can reach is not a measurement of anything served.
#[test]
fn above_range_temperature_is_refused() {
    for bad in [2.5f32, f32::INFINITY] {
        let mut args = unrunnable_args(3);
        args.temperature = bad;
        let msg = run_bench(args)
            .expect_err("out-of-range --temperature must be refused")
            .to_string();
        assert!(
            msg.contains("--temperature"),
            "wrong rejection for {bad}: {msg}"
        );
    }
}

/// The distribution filters sit downstream of the softmax, which the greedy path
/// never builds. Accepting them at temperature 0 records a nucleus or top-k
/// setting against a cell that applied neither — the exact "silently no-opping
/// into a greedy run wearing a sampled label" this validator exists to stop.
#[test]
fn distribution_filters_at_temperature_zero_are_refused() {
    let mut nucleus = unrunnable_args(3);
    nucleus.top_p = 0.9;
    let msg = run_bench(nucleus)
        .expect_err("--top-p at temperature 0 must be refused")
        .to_string();
    assert!(msg.contains("--top-p"), "wrong rejection: {msg}");

    let mut topk = unrunnable_args(3);
    topk.top_k = 20;
    let msg = run_bench(topk)
        .expect_err("--top-k at temperature 0 must be refused")
        .to_string();
    assert!(msg.contains("--top-k"), "wrong rejection: {msg}");

    // ...and both clear the gate once the sampler actually runs.
    let mut sampled = unrunnable_args(3);
    sampled.temperature = 0.7;
    sampled.top_p = 0.95;
    sampled.top_k = 20;
    let msg = run_bench(sampled)
        .expect_err("the nonexistent model path must be what fails")
        .to_string();
    assert!(
        !msg.contains("--top-p") && !msg.contains("--top-k"),
        "filters are legitimate above temperature 0: {msg}"
    );
}

/// A repetition penalty of zero divides positive logits by zero; a negative one
/// flips their sign. Both produce a token stream that is not a penalised one.
#[test]
fn non_positive_repetition_penalty_is_refused() {
    for bad in [0.0f32, -1.0, f32::NAN] {
        let mut args = unrunnable_args(3);
        args.repetition_penalty = bad;
        let msg = run_bench(args)
            .expect_err("non-positive --repetition-penalty must be refused")
            .to_string();
        assert!(
            msg.contains("--repetition-penalty"),
            "wrong rejection for {bad}: {msg}"
        );
    }
}

/// The identity values must pass — a guard that refuses the default would be
/// vacuously "correct" on every test above.
#[test]
fn default_sampler_knobs_pass_validation() {
    let msg = run_bench(unrunnable_args(3))
        .expect_err("the nonexistent model path must be what fails")
        .to_string();
    assert!(
        !msg.contains("--temperature")
            && !msg.contains("--top-p")
            && !msg.contains("--repetition-penalty"),
        "greedy defaults must clear the sampler checks: {msg}"
    );
}

/// The label the summary prints keys off the same condition the decode loop
/// uses to leave the GPU argmax path.
#[test]
fn host_sampled_matches_the_decode_loop_gate() {
    let greedy = unrunnable_args(3);
    assert!(
        !greedy.host_sampled(),
        "temperature 0, no penalty: GPU path"
    );

    let mut sampled = unrunnable_args(3);
    sampled.temperature = 0.7;
    assert!(
        sampled.host_sampled(),
        "temperature > 0 reads logits to host"
    );

    let mut penalised = unrunnable_args(3);
    penalised.repetition_penalty = 1.1;
    assert!(
        penalised.host_sampled(),
        "a penalty reads logits to host even at temperature 0"
    );

    // top_p alone cannot reach this predicate at all: `validate_sampling`
    // refuses the combination first. Were it ever accepted, the cell would be
    // labelled greedy — which is what it would in fact be, since the filters run
    // only inside the temperature path.
    let mut nucleus_only = unrunnable_args(3);
    nucleus_only.top_p = 0.9;
    assert!(
        !nucleus_only.host_sampled(),
        "the filters do not by themselves take the host path"
    );
    assert!(
        nucleus_only.validate_sampling().is_err(),
        "and the combination never reaches a run: it is refused up front"
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

/// Settle a metric the way `summarize` does, so the tests exercise both gates
/// together rather than either one in isolation.
fn settle(name: &str, values_in_order: &[f64]) -> Option<String> {
    let spread = Spread::of(values_in_order).expect("non-empty sample");
    settle_refusal(name, values_in_order, &spread)
}

/// The measured case this guard was built for: gemma-4-e2b at 64k, three
/// consecutive in-process generations, TTFT in milliseconds. `Spread` sorts,
/// so it reports this as a wide range around a median; in collection order it
/// is a ramp, and the median is a point on it.
#[test]
fn measured_ttft_ramp_is_detected_as_drift() {
    let in_order = [17_769.4, 22_764.3, 25_475.1];
    let spread = Spread::of(&in_order).expect("non-empty");
    let pct = detect_drift(&in_order, spread.median).expect("a 43% ramp is a trend");
    assert!(
        pct > 30.0,
        "fitted drift {pct:.1}% understates a 17.8s → 25.5s ramp"
    );
    let msg = settle("ttft_ms", &in_order).expect("a drifting metric must not be summarised");
    assert!(msg.contains("trend, not a spread"), "got: {msg}");
    assert!(msg.contains("rose"), "must name the direction: {msg}");
}

/// Order is the whole point for the *trend* gate: a one-run spike that starts
/// and ends at the same place has slope zero, while the same magnitude of
/// movement spent going one way is a trend.
///
/// A spike is still not a settled cell, though, and at 30% of the median it is
/// nowhere near one — so it is the range gate, not the trend gate, that refuses
/// it. Both must be present: the trend gate alone would pass a spike of any
/// magnitude whatsoever.
#[test]
fn a_spike_is_spread_but_a_ramp_is_drift() {
    let spike = [100.0, 130.0, 100.0];
    let ramp = [100.0, 115.0, 130.0];
    let spike_spread = Spread::of(&spike).expect("non-empty");
    let ramp_spread = Spread::of(&ramp).expect("non-empty");
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
    let spike_msg = settle("ttft_ms", &spike).expect("a 30% spike is not a settled cell");
    assert!(
        spike_msg.contains("never converged"),
        "a spike must be refused by the range gate, not the trend gate: {spike_msg}"
    );
}

/// An unbounded spike is the shape the trend gate cannot see at all: slope
/// stays zero however large it gets, so without a range gate a 40%-wide cell
/// would be summarised as if its median meant something.
#[test]
fn an_arbitrarily_large_spike_is_still_refused() {
    for magnitude in [140.0, 300.0, 10_000.0] {
        let spike = [100.0, magnitude, 100.0];
        assert_eq!(
            detect_drift(&spike, 100.0),
            None,
            "a spike has no slope at any magnitude — {magnitude} must not read as a trend"
        );
        assert!(
            settle("ttft_ms", &spike).is_some(),
            "spike of {magnitude} must still be refused as unsettled"
        );
    }
}

/// Late-onset drift: the fitted change of a single last-run jump shrinks as
/// `--runs` grows (`6d(n−1)/n²` — `1.33d` at 3 runs, `0.29d` at 20), so raising
/// `--runs` in response to a noisy abort would otherwise make the guard weaker.
/// The range gate does not care how many runs the jump is diluted across.
#[test]
fn a_late_onset_step_is_refused_however_many_runs_dilute_it() {
    for n in [3_usize, 10, 20] {
        let mut values: Vec<f64> = std::iter::repeat_n(100.0, n - 1).collect();
        values.push(130.0); // a +30% jump on the final run
        assert!(
            settle("decode_tps", &values).is_some(),
            "a +30% final-run step over {n} runs must be refused"
        );
    }
}

/// Decode TPS falling across runs is the same defect with the opposite sign —
/// the measured Bonsai case, 129.24 → 126.74 → 115.59.
#[test]
fn a_falling_metric_drifts_too() {
    let in_order = [129.24, 126.74, 115.59];
    let spread = Spread::of(&in_order).expect("non-empty");
    let pct = detect_drift(&in_order, spread.median).expect("a 10.6% decline is a trend");
    assert!(pct < 0.0, "direction must be signed");
    let msg = settle("decode_tps", &in_order).expect("a falling metric is still drifting");
    assert!(msg.contains("fell"), "got: {msg}");
}

/// A zig-zag is scatter, not a trend. With three runs the fitted change reduces
/// to `last − first`, so the middle run contributes nothing and this sample
/// fits a −12% "decline" it plainly does not have. Its residuals about the
/// fitted line are twice the size of the change (2×RMS 22.6 vs 12), which is
/// what the noise anchoring exists to notice.
#[test]
fn a_zig_zag_is_scatter_not_a_trend() {
    let in_order = [100.0, 118.0, 88.0];
    assert_eq!(
        detect_drift(&in_order, 100.0),
        None,
        "a sample that jumped up and then further down is not a ramp, however far the \
         endpoints happen to sit apart"
    );
    // It is still not a settled cell — the *range* gate is what refuses it, and
    // it says so as scatter rather than as a decline.
    let msg = settle("decode_tps", &in_order).expect("30% of the median is not settled");
    assert!(msg.contains("never converged"), "got: {msg}");
    assert!(
        !msg.contains("fell"),
        "a zig-zag must not be reported as a direction: {msg}"
    );
}

/// A settled cell must pass. These are the measured gemma-4-e2b ITL p50 values
/// from the same run that produced the TTFT ramp above: the run-to-run noise is
/// under 1%, and refusing it would make the instrument useless.
#[test]
fn a_settled_metric_is_not_drift() {
    let in_order = [9.455, 9.840, 9.538];
    let spread = Spread::of(&in_order).expect("non-empty");
    assert_eq!(detect_drift(&in_order, spread.median), None);
    assert_eq!(settle("itl_p50_ms", &in_order), None);
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
    // Checked through the whole settle decision, not just the trend half — a
    // range gate tightened past these would abort cells that are fine.
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
        assert_eq!(
            settle(name, &v),
            None,
            "{name} is ordinary warmup wobble and must clear both gates"
        );
    }
    // A real ramp from the same model at 64k must still be caught.
    assert!(detect_drift(&[17_065.0, 22_768.2, 24_684.6], 22_768.2).is_some());
    assert!(settle("ttft_ms", &[17_065.0, 22_768.2, 24_684.6]).is_some());
}

/// The range ceiling is calibrated against cells this instrument itself
/// measured as settled: gemma-4-e2b at 64k and Ternary-Bonsai-8B at 4k, 32k and
/// 64k, all `--kv-quant none --runs 3` on an 18-core host carrying a 1-minute
/// load of 3.4-16.3. Every gated metric of a settled cell must clear the
/// ceiling, or the ceiling is refusing measurements rather than protecting them.
///
/// Both groups matter. The tight group is what the instrument looks like when
/// nothing goes wrong; the wide group is the same instrument, same models, on a
/// host that hiccupped once mid-cell — still a settled cell, and the reason the
/// ceiling is not set near the tight group.
#[test]
fn a_settled_cell_clears_the_range_gate() {
    // Tight: most gated metrics of a settled cell live here.
    for (cell, name, v) in [
        (
            "e2b-64k w3",
            "ttft_ms",
            [22_562.683, 22_632.198, 22_776.669],
        ),
        ("e2b-64k w3", "itl_p50_ms", [9.855, 9.942, 9.998]),
        ("bonsai-4k w0", "decode_tps", [137.677, 138.533, 136.919]),
        ("bonsai-4k w1", "itl_p50_ms", [7.122, 6.980, 7.128]),
        (
            "bonsai-32k w2",
            "ttft_ms",
            [19_510.139, 19_522.153, 19_543.872],
        ),
        ("bonsai-32k w2", "decode_tps", [66.474, 65.434, 65.330]),
        (
            "bonsai-64k w0",
            "ttft_ms",
            [58_050.704, 59_936.437, 59_895.033],
        ),
        ("bonsai-64k w0", "decode_tps", [39.640, 38.301, 39.072]),
    ] {
        let spread = Spread::of(&v).expect("non-empty");
        assert!(
            spread.range_pct() < SETTLED_MAX_RANGE_PCT / 4.0,
            "{cell} {name} is a tight settled cell and must clear the \
             {SETTLED_MAX_RANGE_PCT}% ceiling with at least 4x margin, observed range {:.2}%",
            spread.range_pct()
        );
        assert_eq!(settle(name, &v), None, "{cell} {name} must not be refused");
    }

    // Wide but still settled: one run of the cell caught a host hiccup. These
    // are what stops the ceiling from being set near the tight group — at 10%
    // they would have had ~1.2x of margin.
    for (cell, name, v) in [
        ("bonsai-4k w1", "ttft_ms", [1204.071, 1300.797, 1279.021]),
        ("e2b-64k w3", "decode_tps", [99.327, 99.607, 92.561]),
    ] {
        let spread = Spread::of(&v).expect("non-empty");
        assert!(
            spread.range_pct() > 7.0 && spread.range_pct() < SETTLED_MAX_RANGE_PCT,
            "{cell} {name} is the wide tail of the settled population, observed range {:.2}%",
            spread.range_pct()
        );
        assert_eq!(
            settle(name, &v),
            None,
            "{cell} {name} settled and must not be refused"
        );
    }
}

/// A settled cell's `itl_p99_ms`: 58.6% of range on a gemma-4-e2b 64k cell
/// whose every gated metric ranged under 2%. This is the measurement behind
/// [`Gate::Ungated`] for p99 — gating it would abort settled cells, and no
/// plausible ceiling separates it from a real defect.
#[test]
fn the_tail_statistic_would_abort_settled_cells_if_it_were_gated() {
    let p99 = [16.337, 10.261, 10.370]; // e2b-64k --warmup 3
    let spread = Spread::of(&p99).expect("non-empty");
    assert!(
        spread.range_pct() > SETTLED_MAX_RANGE_PCT,
        "p99 ranged {:.1}% on a settled cell — the reason it is not gated",
        spread.range_pct()
    );
    assert_eq!(Metric::ItlP99Ms.gate(), Gate::Ungated);
}

/// KV bytes are bit-identical across the runs of a cell, so a range gate that
/// keyed off anything but the observed values would show up here first.
#[test]
fn a_constant_metric_clears_both_gates() {
    let in_order = [649_353_216.0, 649_353_216.0, 649_353_216.0];
    assert_eq!(settle("kv_cache_bytes", &in_order), None);
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
    let msg = assert_one_token_stream(&[
        ("measured run 1".to_owned(), a),
        ("measured run 2".to_owned(), b),
    ])
    .expect_err("runs of one cell must decode the same tokens")
    .to_string();
    assert!(msg.contains("different token streams"), "got: {msg}");
    assert!(assert_one_token_stream(&[
        ("measured run 1".to_owned(), a),
        ("measured run 2".to_owned(), a),
    ])
    .is_ok());
}

/// A cold-start defect corrupts run 1, not runs 2-N. Anchoring on the first run
/// would report every later run as the deviant one and send the operator after
/// the wrong runs; the majority is what identifies the outlier.
#[test]
fn the_outlier_run_is_named_not_the_majority() {
    let good = token_stream_digest([1_u32, 2, 3]);
    let cold = token_stream_digest([9_u32, 9, 9]);
    let msg = assert_one_token_stream(&[
        ("warmup run 1".to_owned(), cold),
        ("measured run 1".to_owned(), good),
        ("measured run 2".to_owned(), good),
        ("measured run 3".to_owned(), good),
    ])
    .expect_err("a cell with two different streams is not one cell")
    .to_string();
    assert!(
        msg.contains(&format!("Most common digest {good:#018x} (3 of 4 run(s))")),
        "the majority stream must be named as the reference: {msg}"
    );
    // Every run is listed, and only the cold one is marked.
    assert_eq!(
        msg.matches("<-- differs").count(),
        1,
        "exactly the outlier is marked: {msg}"
    );
    assert!(
        msg.contains("warmup run 1")
            && msg.contains("measured run 1")
            && msg.contains("measured run 3"),
        "every run must be listed with its digest: {msg}"
    );
}

/// With two runs and two digests there is no majority. The instrument must
/// still name both rather than silently picking one as correct.
#[test]
fn a_two_run_disagreement_names_both() {
    let a = token_stream_digest([1_u32]);
    let b = token_stream_digest([2_u32]);
    let msg = assert_one_token_stream(&[
        ("measured run 1".to_owned(), a),
        ("measured run 2".to_owned(), b),
    ])
    .expect_err("two different streams are not one cell")
    .to_string();
    assert!(msg.contains(&format!("{a:#018x}")), "got: {msg}");
    assert!(msg.contains(&format!("{b:#018x}")), "got: {msg}");
    assert!(
        msg.contains("1 of 2 run(s)"),
        "no majority is stated: {msg}"
    );
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

// ── Which metrics are gated is enumerated, not positional ───────────

/// One run's worth of sample, with every metric settable.
fn run_sample(ttft_ms: f64, decode_tps: f64, kv: u64) -> RunSample {
    RunSample {
        ttft_ms,
        decode_tps,
        prefill_tps: Some(1000.0 / ttft_ms),
        itl_p50_ms: 10.0,
        itl_p99_ms: 12.0,
        kv_cache_bytes: kv,
        n_generated: 128,
        token_digest: 0xabc,
    }
}

/// The gate decision is a total function of the metric. This does not merely
/// re-state the match — it pins that both lists are non-empty, so a change that
/// gated everything (aborting settled cells on p99 noise) or gated nothing
/// (the state this guard replaced) fails here.
#[test]
fn every_metric_names_a_gate_and_both_lists_are_populated() {
    let gated: Vec<&str> = Metric::ALL
        .iter()
        .filter(|m| m.gate() == Gate::Settled)
        .map(|m| m.name())
        .collect();
    let ungated: Vec<&str> = Metric::ALL
        .iter()
        .filter(|m| m.gate() == Gate::Ungated)
        .map(|m| m.name())
        .collect();
    assert_eq!(
        gated,
        vec!["ttft_ms", "itl_p50_ms", "decode_tps", "kv_cache_bytes"],
        "the metrics a decision is made on must all be gated"
    );
    assert_eq!(
        ungated,
        vec!["itl_p99_ms", "prefill_tps"],
        "p99 is an extreme-order statistic and prefill_tps is a reciprocal of ttft_ms; \
         both are reported, neither is evidence"
    );
    assert_eq!(
        gated.len() + ungated.len(),
        Metric::ALL.len(),
        "every metric is in exactly one list"
    );
}

/// `prefill_tps` is `prompt_tokens / ttft`, with `prompt_tokens` identical in
/// every run of a cell — so gating it would test `ttft_ms` twice under a
/// nonlinear transform, and the transform is what decides whether they agree.
/// Its verdict must not be able to abort a cell, whatever it says on its own.
#[test]
fn prefill_tps_cannot_abort_a_cell() {
    assert_eq!(Metric::PrefillTps.gate(), Gate::Ungated);
    // A blatant ramp in prefill_tps, on runs whose gated metrics all settled.
    let mut samples = [
        run_sample(1000.0, 100.0, 4096),
        run_sample(1000.0, 100.0, 4096),
        run_sample(1000.0, 100.0, 4096),
    ];
    samples[0].prefill_tps = Some(500.0);
    samples[1].prefill_tps = Some(750.0);
    samples[2].prefill_tps = Some(1000.0);
    let prefill = [500.0, 750.0, 1000.0];
    assert!(
        settle("prefill_tps", &prefill).is_some(),
        "the values themselves are a ramp — the point is that nothing asks"
    );
    let summary = summarize(&samples).expect("gated metrics settled, so the cell stands");
    let reported = summary
        .prefill_tps
        .expect("dropping the gate must not drop the reporting");
    assert!((reported.median - 750.0).abs() < 1e-9);
    assert!(reported.range_pct() > 60.0, "its spread is still visible");
}

/// A contended run moves more than one metric. Refusing at the first hides how
/// much of the cell moved, which is the difference between "one metric wobbled"
/// and "the whole run was contended".
#[test]
fn every_unsettled_metric_is_named_not_just_the_first() {
    // ttft ramps, decode TPS falls, kv bytes stay put.
    let samples = [
        run_sample(1000.0, 130.0, 4096),
        run_sample(1400.0, 126.0, 4096),
        run_sample(1800.0, 112.0, 4096),
    ];
    let msg = summarize(&samples)
        .expect_err("a cell that moved in two metrics is not summarisable")
        .to_string();
    assert!(msg.contains("ttft_ms"), "got: {msg}");
    assert!(msg.contains("decode_tps"), "got: {msg}");
    assert!(
        msg.contains("2 of its 4 gated metric(s)"),
        "the count of moved metrics is what says how bad it was: {msg}"
    );
}
