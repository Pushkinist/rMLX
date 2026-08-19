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
    ];
    for name in spec_names {
        assert!(
            METRICS.iter().any(|(n, _, _, _)| *n == name),
            "metric '{name}' missing from METRICS"
        );
    }
    // METRICS row count — bump when adding new metric ops.
    assert_eq!(METRICS.len(), 55, "METRICS should have exactly 55 rows");
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

/// COVERAGE_MATRIX must have at least one row for each of the 5 spec backends.
#[test]
fn coverage_matrix_includes_all_5_backends() {
    let required = ["rmlx", "mlx_lm", "paroquant", "omlx", "ollama"];
    for backend in required {
        assert!(
            COVERAGE_MATRIX.iter().any(|(b, _, _)| *b == backend),
            "backend '{backend}' has no rows in COVERAGE_MATRIX"
        );
    }
}
