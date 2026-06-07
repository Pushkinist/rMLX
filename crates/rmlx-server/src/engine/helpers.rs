//! Shared helper functions used by both Gemma4Generator and SpeculativeGenerator.
//!
//! - ITL / TTFT events-table write helpers
//! - `resolve_kv_quant_for_load` — auto-kv-quant resolution
//! - `is_reconstructible_tool_marker` — Gemma-4 tool protocol marker allowlist
//! - `compute_itl_stats` — ITL percentile / mean / spike computation
//! - `spsc_ts` — UTC timestamp string for SPSC metric events
//! - `kv_quant_label` — KvQuant → label string

use std::time::Instant;

use rmlx_metrics::events::{EventRecorder, Measurement};

// ── shared helper for ITL percentile events-table writes ───────────────

/// Write p50/p95/p99 ITL measurements to the events table.
///
/// Called from both Gemma4Generator and SpeculativeGenerator after decode
/// completes. A single helper avoids six near-identical `rec.record` blocks.
///
/// Also writes the equivalent `tpot_*_ms` rows. In v1 the values are
/// numerically identical to the `itl_*_ms` rows because both definitions
/// exclude the first interval (= pure decode-only intervals); the new
/// `tpot_*` ops are emitted under a separate name so downstream queries can
/// adopt the TPOT convention now and the two op families can diverge later
/// (e.g. `tpot_*` may one day exclude tool-calling stalls). The legacy
/// `itl_*` rows are kept for backward compat.
pub(crate) fn record_itl_percentiles(
    rec: &EventRecorder,
    model_id: &str,
    quant: &str,
    p50: f64,
    p95: f64,
    p99: f64,
) {
    for (op, v) in [
        ("itl_p50_ms", p50),
        ("itl_p95_ms", p95),
        ("itl_p99_ms", p99),
        // TPOT mirrors ITL numerically in v1; emitted as its own
        // canonical metric name for stage-attribution clarity.
        ("tpot_p50_ms", p50),
        ("tpot_p95_ms", p95),
        ("tpot_p99_ms", p99),
    ] {
        if let Err(e) = rec.record(&Measurement {
            model_path: model_id,
            quant_mode: quant,
            stage: "request",
            op,
            value_unit: "ms",
            value: v,
            notes: "",
        }) {
            tracing::warn!(error = %e, op, "events-table write failed");
        }
    }
}

// ── shared helper for TTFT + prefill_duration_ms events-table writes ───

/// Write the TTFT row + the `prefill_duration_ms` row in one call.
/// Both share a single timestamp (no drift between them).
///
/// Caller must invoke from a blocking context (e.g. inside `spawn_blocking`)
/// — this function does synchronous SQLite I/O.
pub(crate) fn record_ttft_and_prefill(
    rec: &EventRecorder,
    model_id: &str,
    is_cold: bool,
    ttft_ms: u64,
) {
    let ttft_op = if is_cold {
        "ttft_cold_ms"
    } else {
        "ttft_warm_ms"
    };
    let ttft_ms_f = ttft_ms as f64;
    for (op, v) in [(ttft_op, ttft_ms_f), ("prefill_duration_ms", ttft_ms_f)] {
        if let Err(e) = rec.record(&Measurement {
            model_path: model_id,
            quant_mode: "n/a",
            stage: "request",
            op,
            value_unit: "ms",
            value: v,
            notes: "",
        }) {
            tracing::warn!(error = %e, op, "events-table write failed");
        }
    }
}

// ── shared model-load configuration helper ───────────────────────────────────

/// Resolve `--kv-quant=auto` (i.e. `None`) against a loaded `config.json`.
///
/// The two generator constructors carried this identical ~20-line block
/// twice. Extracted to a single home so the per-arch resolver
/// and the `user_explicit` tracking live in exactly one place.
///
/// Returns `(resolved_quant, user_explicit)`:
/// - explicit override (`Some`) → `(Some(q), true)`.
/// - auto (`None`) → resolve via the per-arch default table → `(Some(r), false)`.
///
/// `user_explicit=false` lets per-request `kv_quant_for_ctx` override in auto mode.
pub(crate) fn resolve_kv_quant_for_load(
    cfg: &rmlx_loader::ModelConfig,
    kv_quant: Option<rmlx_kv_quant::KvQuant>,
    model_id: &str,
) -> (Option<rmlx_kv_quant::KvQuant>, bool) {
    if let Some(q) = kv_quant {
        (Some(q), true)
    } else {
        let arch_class = cfg.architectures.first().map_or("(empty)", String::as_str);
        let signals = rmlx_models::kv_cache::ResolverSignals::from_config(cfg);
        let resolved = rmlx_models::kv_cache::KvCacheBuilder::resolve_default(arch_class, signals);
        tracing::info!(
            model_id = %model_id,
            arch = arch_class,
            hidden_size = ?signals.hidden_size,
            has_moe = signals.has_moe,
            is_paroquant = signals.is_paroquant,
            weight_bits = ?signals.weight_bits,
            ?resolved,
            "kv-quant=auto resolved via per-arch default table"
        );
        (Some(resolved), false)
    }
}

/// Issue #26: parse a per-request `kv_quant` string into an optional override.
///
/// Mirrors the `--kv-quant` CLI grammar (`crate`-external
/// `rmlx_cli::commands::parse::parse_kv_quant`) so a request field and the
/// launch flag accept identical strings:
/// - `"auto"` → `Ok(None)` — fall through to the generator's per-arch/per-ctx
///   auto policy (NOT the launch explicit value).
/// - `"mixed"` → the canonical `Mixed{k8,v4,g64}` short alias.
/// - everything else → `<KvQuant as FromStr>::from_str` (`none`/`bf16`,
///   `k8v4`, `k8v8`, `planar`, `mixed_k<kb>g<kg>_v<vb>g<vg>`, …).
///
/// Returns the parse error string (no `--kv-quant:` prefix) so the route layer
/// can wrap it in a clean HTTP 400 `invalid_request_error`.
pub(crate) fn parse_request_kv_quant(s: &str) -> Result<Option<rmlx_kv_quant::KvQuant>, String> {
    use rmlx_kv_quant::KvQuant;
    use std::str::FromStr;
    let s = s.trim();
    if s.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    if s.eq_ignore_ascii_case("mixed") {
        return Ok(Some(KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }));
    }
    KvQuant::from_str(s).map(Some).map_err(|e| e.to_string())
}

// ── tool-marker allowlist ────────────────────────────────────────────────────

/// A5.6: whether a special-token surface form is one of the Gemma-4
/// tool-call protocol markers the response parser needs reconstructed.
///
/// Restricted to exactly the three markers the [`crate::tool_parser`]
/// `GemmaToolCall` path consumes — `<|tool_call>`, `<tool_call|>`, `<|"|>`.
/// These never appear outside a tool call, so reconstructing them into the
/// decoded stream cannot pollute visible content. Other Gemma specials
/// (`<turn|>`, `<|channel>`, `<|tool>`, …) are deliberately excluded so
/// they stay suppressed exactly as before this change.
pub(crate) fn is_reconstructible_tool_marker(surface: &str) -> bool {
    matches!(surface, "<|tool_call>" | "<tool_call|>" | "<|\"|>")
}

/// Test-only accessor so `tool_parser` unit tests can assert the engine's
/// marker allowlist stays tight (no `<turn|>` / `<|channel>` leakage).
#[cfg(test)]
pub(crate) fn tests_support_is_reconstructible_tool_marker(surface: &str) -> bool {
    is_reconstructible_tool_marker(surface)
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Compute ITL p50/p95/p99/mean/spikes (in ms) from per-step `Instant` timestamps.
///
/// Returns `None` when fewer than 2 timestamps are present (no interval can
/// be computed from a single point). Intervals are the gaps between
/// consecutive arrivals — i.e. `timestamps[i+1] - timestamps[i]`.
///
/// p50, p95, p99 are computed via nearest-rank (lower interpolation) on the
/// sorted interval list. Implementation is allocation-minimal: one `Vec<f64>`
/// of length `n - 1`, sorted in-place.
///
/// # Spike count definition
///
/// A spike is any interval strictly greater than `3 × median (p50)`. The
/// 3× median threshold was chosen over mean-based thresholds because the median
/// is robust to the very outliers we are trying to count — a mean-based
/// threshold (`mean + 3·stddev`) degrades under heavy spikes where the mean
/// itself is elevated. Threshold documented here per CLAUDE.md "document the
/// truth" rule.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
pub(crate) fn compute_itl_stats(timestamps: &[Instant]) -> Option<(f64, f64, f64, f64, u64)> {
    let n = timestamps.len();
    if n < 2 {
        return None;
    }

    // Build sorted interval list (ms).
    let mut intervals: Vec<f64> = timestamps
        .windows(2)
        .map(|w| w[1].duration_since(w[0]).as_secs_f64() * 1000.0)
        .collect();
    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let count = intervals.len() as f64;
    let mean = intervals.iter().sum::<f64>() / count;

    // Percentile via nearest-rank (lower interpolation).
    let percentile = |pct: f64| -> f64 {
        let idx = (pct / 100.0).mul_add(count, -1.0).ceil().max(0.0) as usize;
        intervals[idx.min(intervals.len() - 1)]
    };

    let p50 = percentile(50.0);
    let p95 = percentile(95.0);
    let p99 = percentile(99.0);

    // Spike count: intervals exceeding 3× median (spike threshold = 3 × p50).
    let spike_threshold = 3.0 * p50;
    let spikes = intervals.iter().filter(|&&v| v > spike_threshold).count() as u64;

    Some((p50, p95, p99, mean, spikes))
}

/// Return an ISO-8601 UTC timestamp string for SPSC metric events.
pub(crate) fn spsc_ts() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

/// Map a `KvQuant` override to its canonical label string.
pub(crate) fn kv_quant_label(kv: Option<rmlx_kv_quant::KvQuant>) -> String {
    match kv {
        Some(rmlx_kv_quant::KvQuant::K8V8) => "k8v8",
        Some(rmlx_kv_quant::KvQuant::K8V4) => "k8v4",
        Some(rmlx_kv_quant::KvQuant::Planar) => "planar",
        Some(rmlx_kv_quant::KvQuant::Planar3) => "planar3",
        Some(rmlx_kv_quant::KvQuant::None) => "none",
        Some(rmlx_kv_quant::KvQuant::Mixed { .. }) => "mixed",
        Some(rmlx_kv_quant::KvQuant::RotK { .. }) => "rot_k",
        Some(rmlx_kv_quant::KvQuant::RotKTq4V) => "rot_k_tq4v",
        Some(rmlx_kv_quant::KvQuant::K8VTurbo3) => "k8vturbo3",
        Some(rmlx_kv_quant::KvQuant::TurboSym4) => "tsym4",
        Some(rmlx_kv_quant::KvQuant::PlanarK) => "planar_k",
        Some(rmlx_kv_quant::KvQuant::K8VTurbo2) => "k8vturbo2",
        Some(rmlx_kv_quant::KvQuant::Iso3) => "iso3",
        Some(rmlx_kv_quant::KvQuant::Iso4) => "iso4",
        Some(rmlx_kv_quant::KvQuant::Rotor3) => "rotor3",
        Some(rmlx_kv_quant::KvQuant::Rotor4) => "rotor4",
        Some(rmlx_kv_quant::KvQuant::K8VTurbo3Tcq) => "k8vturbo3tcq",
        Some(rmlx_kv_quant::KvQuant::K8VTurbo2Tcq) => "k8vturbo2tcq",
        Some(rmlx_kv_quant::KvQuant::Iso3Sym) => "iso3_sym",
        Some(rmlx_kv_quant::KvQuant::Iso4Sym) => "iso4_sym",
        Some(rmlx_kv_quant::KvQuant::IsoKOnly3) => "k_iso3",
        Some(rmlx_kv_quant::KvQuant::IsoKOnly4) => "k_iso4",
        Some(rmlx_kv_quant::KvQuant::Rotor3Sym) => "rotor3_sym",
        Some(rmlx_kv_quant::KvQuant::Rotor4Sym) => "rotor4_sym",
        Some(rmlx_kv_quant::KvQuant::RotorKOnly3) => "k_rotor3",
        Some(rmlx_kv_quant::KvQuant::RotorKOnly4) => "k_rotor4",
        // Payload-bearing asymmetric rotor-K variants — render via
        // Display so the v-side spec is captured (`rotor_k_*_asym_v*_g*`).
        Some(
            kq @ (rmlx_kv_quant::KvQuant::RotorK3Asym { .. }
            | rmlx_kv_quant::KvQuant::RotorK4Asym { .. }),
        ) => return format!("{kq}"),
        // TurboSym3 — symmetric WHT-3 K+V.
        Some(rmlx_kv_quant::KvQuant::TurboSym3) => "tsym3",
        None => "auto",
    }
    .into()
}
