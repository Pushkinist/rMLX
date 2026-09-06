use super::*;

/// All spec metric names must be present in METRICS.
#[test]
fn every_spec_metric_present() {
    let spec_names = [
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
        // N19 additions.
        "prompt_cache_hits",
        "prompt_cache_misses",
        "prompt_cache_bytes",
        // C6 block-level counters.
        "prompt_cache_block_hits",
        "prompt_cache_block_misses",
        "prompt_cache_partial_hits",
        "prompt_cache_hot_cache_hits",
        "prompt_cache_hot_cache_evictions",
        // SSD-tier hydrate hits.
        "prompt_cache_ssd_hits",
        // Per-phase load-time spans.
        "load_mmap_ms",
        "load_dequant_ms",
        "load_gpu_residency_ms",
        "load_first_kernel_ready_ms",
        "load_total_ms",
        // C5 Slice A: admission-queue metrics.
        "queue_wait_ms",
        "queue_depth",
        // F1b: per-request live token counts.
        "prompt_tokens_live",
        "completion_tokens_live",
        // F9: extended ITL stats.
        "itl_p99_ms",
        "itl_spikes",
        // Speculative-decoding metrics.
        "accept_rate",
        "draft_tokens_total",
        "accept_tokens_total",
        "draft_rounds_total",
        "accepted_per_step",
        "tokens_per_round",
        "draft_ms_per_round",
        "verify_ms_per_round",
        "loop_ms_per_round",
        // SSD-tier observability (step2).
        "ssd_bytes_used",
        "ssd_evict_total",
        // Raw per-event latency observations (H2: p50/p99 dropped — single-sample
        // percentiles are meaningless; real aggregation via Prometheus histogram).
        "ssd_spill_ms",
        "ssd_hydrate_ms",
        "ssd_spill_mb_per_s",
        "ssd_hydrate_mb_per_s",
        // phase-split TTFT/TPOT metrics.
        "prefill_duration_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        // one per `rmlx eval ppl` scorer: the cacheless full-window forward and
        // the cache-bearing teacher-forced one do not measure the same thing.
        "ppl_wikitext2_cached",
        // resident memory has two counters and they are not the same number
        "peak_phys_footprint_mb",
    ];
    for name in spec_names {
        assert!(
            METRICS.iter().any(|(n, _, _, _)| *n == name),
            "metric '{name}' missing from METRICS"
        );
    }
    // METRICS row count — bump when adding new metric ops.
    assert_eq!(METRICS.len(), 61, "METRICS should have exactly 61 rows");
}

#[test]
fn lookup_known_returns_unit_and_direction() {
    let (unit, dir) = lookup("decode_tps_warm").expect("decode_tps_warm must be in registry");
    assert_eq!(unit, "tps");
    assert_eq!(dir, Direction::HigherBetter);
}

#[test]
fn lookup_unknown_errors() {
    let err = lookup("foo").unwrap_err();
    assert!(
        matches!(err, Error::UnknownMetric(ref s) if s == "foo"),
        "expected UnknownMetric(\"foo\"), got {err:?}"
    );
}

#[test]
fn direction_roundtrip() {
    assert_eq!(
        Direction::parse("higher_better").unwrap(),
        Direction::HigherBetter
    );
    assert_eq!(
        Direction::parse("lower_better").unwrap(),
        Direction::LowerBetter
    );
    assert!(Direction::parse("nope").is_err());
}

/// parse is strict — mixed-case must not match.
#[test]
fn direction_str_lowercase_only() {
    assert!(Direction::parse("Higher_Better").is_err());
    assert!(Direction::parse("LOWER_BETTER").is_err());
    assert!(Direction::parse("Higher_better").is_err());
}

/// Every unit string must be one of the known SI-friendly units.
#[test]
fn every_metric_has_valid_unit() {
    // added the `ppl` (perplexity) and `nat` (natural-log nats / NLL) units.
    let valid_units = [
        "tps", "ms", "mb", "bytes", "ratio", "count", "mb/s", "ppl", "nat",
    ];
    for (name, unit, _, _) in METRICS {
        assert!(
            valid_units.contains(unit),
            "metric '{name}' has unknown unit '{unit}'"
        );
    }
}

#[test]
fn coverage_lookup_known_pair() {
    assert_eq!(coverage("rmlx", "decode_tps_warm"), Coverage::Yes);
}

#[test]
fn coverage_lookup_unknown_pair_is_no() {
    assert_eq!(coverage("xyz", "decode_tps_warm"), Coverage::No);
}

/// Every whitelisted backend must be wired into `COVERAGE_MATRIX`, or declared
/// in `BACKENDS_WITHOUT_COVERAGE`.
///
/// The previous version of this test hard-coded "the 5 spec backends", so a
/// backend added to the whitelist afterwards landed half-wired and silently:
/// `coverage()` falls back to `No` for an unknown pair, which is
/// indistinguishable from a measured "this backend cannot emit that metric".
/// `llama_cpp` and `llama_cpp_tq` both reached the whitelist that way. Driving
/// the sweep off the whitelist itself is what makes the next one impossible.
#[test]
fn coverage_matrix_covers_every_whitelisted_backend() {
    for backend in crate::identity::BACKEND_WHITELIST {
        let has_rows = COVERAGE_MATRIX.iter().any(|(b, _, _)| b == backend);
        let declared = BACKENDS_WITHOUT_COVERAGE.contains(backend);
        assert!(
            has_rows != declared,
            "backend '{backend}': has_rows={has_rows}, declared_uncovered={declared} \
             — it must be exactly one. Add its COVERAGE_MATRIX rows, or add it to \
             BACKENDS_WITHOUT_COVERAGE with the reason."
        );
    }
}

/// A backend that is wired must be wired for the *whole* metric spec, not a
/// convenient subset — otherwise a missing cell reads as `No` by fallback.
#[test]
fn wired_backends_declare_every_spec_metric() {
    for backend in crate::identity::BACKEND_WHITELIST {
        if BACKENDS_WITHOUT_COVERAGE.contains(backend) {
            continue;
        }
        for metric in BACKEND_METRIC_SPEC {
            assert!(
                COVERAGE_MATRIX
                    .iter()
                    .any(|(b, m, _)| b == backend && m == metric),
                "backend '{backend}' has no COVERAGE_MATRIX row for spec metric \
                 '{metric}'; `coverage()` would answer No by fallback, which is \
                 not the same as a measured No."
            );
        }
    }
}

// ── Bounds ────────────────────────────────────────────────────────────────

#[test]
fn bounds_reject_negatives_whatever_the_floor() {
    assert!(!Bounds::non_negative(10.0).contains(-1.0));
    assert!(!Bounds::positive(10.0).contains(-1.0));
    assert!(!Bounds::non_negative(10.0).contains(-0.000_001));
}

#[test]
fn bounds_ceiling_is_inclusive() {
    assert!(Bounds::positive(10.0).contains(10.0));
    assert!(!Bounds::positive(10.0).contains(10.1));
    assert!(Bounds::non_negative(10.0).contains(10.0));
    assert!(!Bounds::non_negative(10.0).contains(f64::MAX));
}

#[test]
fn bounds_floor_differs_by_constructor() {
    assert!(!Bounds::positive(10.0).contains(0.0));
    assert!(Bounds::positive(10.0).contains(f64::MIN_POSITIVE));
    assert!(Bounds::non_negative(10.0).contains(0.0));
}

#[test]
fn bounds_reject_non_finite() {
    for b in [Bounds::positive(10.0), Bounds::non_negative(10.0)] {
        assert!(!b.contains(f64::NAN));
        assert!(!b.contains(f64::INFINITY));
        assert!(!b.contains(f64::NEG_INFINITY));
    }
}

/// The SQL rendering is what the `bests` view and the `deltas`/`timeseries`
/// queries actually enforce, so pin the string, not just its shape.
#[test]
fn bounds_render_the_floor_they_declare() {
    assert_eq!(
        Bounds::positive(1e5).sql("value"),
        "value > 0.0 AND value <= 100000.0"
    );
    assert_eq!(
        Bounds::non_negative(1e12).sql("value"),
        "value >= 0.0 AND value <= 1000000000000.0"
    );
}

#[test]
fn bounds_describe_shows_the_open_end() {
    assert_eq!(Bounds::positive(1e5).describe(), "(0, 100000.0]");
    assert_eq!(Bounds::non_negative(1.0).describe(), "[0, 1.0]");
}

/// Both PPL scorers' metrics are registered, with the same unit and direction.
///
/// An unregistered name is refused at `RunRecord::validate`, so the run's
/// number never reaches the store — the command still exits 0 and leaves a
/// buffer file, which is exactly what happened to the first eight cache-bearing
/// sweeps. A metric the CLI can emit and the registry does not know is a
/// measurement thrown away quietly.
#[test]
fn both_ppl_scorers_have_a_registered_metric() {
    let (cacheless_unit, cacheless_dir) =
        lookup("ppl_wikitext2").expect("the cacheless scorer's metric");
    let (cached_unit, cached_dir) =
        lookup("ppl_wikitext2_cached").expect("the cached scorer's metric");
    assert_eq!(
        (cacheless_unit, cacheless_dir),
        (cached_unit, cached_dir),
        "the two scorers measure perplexity in the same unit and rank the same way; \
         only what produced the number differs"
    );
}
