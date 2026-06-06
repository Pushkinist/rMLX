//! Decode-loop scaffolding shared across architectures.
//!
//! # Status
//!
//! Stage 1 of the runtime extraction surfaces *types* and *small helpers*
//! that callers of `generate_greedy` use repeatedly. The full outer-loop
//! refactor (chunked prefill + decode + decode-profile timers) is deferred
//! until at least three archs have migrated to the runtime helpers
//! provided here. Reason: the existing per-arch `generate_greedy` functions
//! all return `crate::gemma4::ProbeStep` today, so introducing a
//! `runtime::ProbeStep` and rewriting one arch's `generate_greedy` would
//! force a cross-arch type rename that the task explicitly forbids
//! ("no changes to non-migrated arches").
//!
//! What is provided here right now:
//! - [`ProbeStep`], [`SmokeVerdict`], [`DecodeProfile`] — runtime-native
//!   types that future arches and the unified `generate_greedy` will use.
//!   Structurally identical to `crate::gemma4::ProbeStep` /
//!   `crate::gemma4::SmokeVerdict`. They co-exist for now and the gemma4
//!   versions will be migrated to type aliases in a follow-up.
//! - [`PREFILL_CHUNK`] — the standard 64-token prefill chunk size (Metal
//!   watchdog safety margin).
//! - [`DecodeProfile::log`] — emits the standard `decode_profile` tracing
//!   event used by decode-loop performance analysis.
//!
//! See `migration.md` (in the crate root) for the per-arch migration recipe.

use std::time::Instant;

/// Standard prefill chunk size (tokens). Empirically safe under the Metal
/// command-buffer watchdog (~10s budget) for all 6 supported archs at
/// max-ctx 4k–8k. Lower means more sync points and lower prefill TPS.
pub const PREFILL_CHUNK: usize = 64;

// ---------------------------------------------------------------------------
// ProbeStep / SmokeVerdict — structurally identical to gemma4 versions.
// ---------------------------------------------------------------------------

/// One record per autoregressive step.
///
/// Mirrors [`rmlx_models::gemma4::ProbeStep`] field-for-field. Once all six
/// arch graphs migrate to the runtime crate, the gemma4 version becomes a
/// type alias.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — per-step probe record; adding a field requires updating all generate_greedy implementations and smoke classifiers"
)]
#[derive(Debug, Clone)]
pub struct ProbeStep {
    /// Token id produced at this step.
    pub token_id: u32,
    /// Decoded text piece for this token.
    pub piece: String,
    /// Maximum absolute logit value before sampling.
    pub max_abs_logit: f32,
    /// Number of NaN values detected in the logit buffer.
    pub nan_count: usize,
}

/// Verdict returned by `classify_smoke`. Mirrors
/// [`rmlx_models::gemma4::SmokeVerdict`] variant-for-variant.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — four smoke outcomes; adding an outcome requires updating classify_smoke and all verdict consumers"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmokeVerdict {
    /// Generation passed all smoke checks.
    Ok,
    /// Model is looping on punctuation or a single repeated token.
    BrokenPunctLoop {
        /// The piece that dominated the loop.
        dominant_piece: String,
        /// Number of distinct token ids seen during the loop.
        distinct_ids: usize,
    },
    /// NaN was detected in logits at the given step.
    BrokenNan {
        /// Decode step index at which the first NaN appeared.
        at_step: usize,
    },
    /// Outcome cannot be determined with the available budget.
    Inconclusive {
        /// Human-readable reason why the verdict is inconclusive.
        reason: String,
    },
}

// ---------------------------------------------------------------------------
// DecodeProfile — accumulator for the standard decode-loop timer set.
// ---------------------------------------------------------------------------

/// Coarse decode-loop timers. Each per-arch `generate_greedy`
/// currently maintains these as four `u128` locals; this struct centralises
/// the bookkeeping and the final `tracing::info!` emission so future
/// migrations only need to call `start_prefill`, `tick_step`, and `log`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — five decode-loop timer fields; adding a timer requires updating record_step, log, and all arch generate_greedy callers"
)]
#[derive(Debug, Default)]
pub struct DecodeProfile {
    /// Cumulative nanoseconds spent in the forward dispatch across all decode steps.
    pub forward_total_ns: u128,
    /// Cumulative nanoseconds spent in eval/argmax/to_bytes across all decode steps.
    pub eval_total_ns: u128,
    /// Cumulative nanoseconds for full step wall-clock (forward through piece resolved).
    pub step_total_ns: u128,
    /// Nanoseconds elapsed during the prefill phase.
    pub prefill_total_ns: u128,
    /// Number of decode steps recorded.
    pub decode_steps: u32,
}

impl DecodeProfile {
    /// Start a fresh prefill window. Returns an `Instant` the caller stamps
    /// at the end of prefill via `record_prefill`.
    pub fn start_prefill() -> Instant {
        Instant::now()
    }

    /// Record the elapsed prefill time.
    pub fn record_prefill(&mut self, t0: Instant) {
        self.prefill_total_ns = t0.elapsed().as_nanos();
    }

    /// Record one decode step. Pass three nanosecond deltas:
    /// `forward_dt` — forward dispatch wall-clock
    /// `eval_dt` — eval/argmax/to_bytes wall-clock
    /// `step_dt` — full step wall-clock (forward .. piece resolved)
    pub fn record_step(&mut self, forward_dt: u128, eval_dt: u128, step_dt: u128) {
        self.forward_total_ns += forward_dt;
        self.eval_total_ns += eval_dt;
        self.step_total_ns += step_dt;
        self.decode_steps += 1;
    }

    /// Emit the standard `decode_profile` tracing event. `arch` should be
    /// the architecture string (e.g. `"Gemma3ForConditionalGeneration"`)
    /// used to filter decode-loop performance events.
    pub fn log(&self, arch: &'static str) {
        let prefill_ms = (self.prefill_total_ns as f64) / 1.0e6;
        let forward_ms = (self.forward_total_ns as f64) / 1.0e6;
        let eval_ms = (self.eval_total_ns as f64) / 1.0e6;
        let step_ms = (self.step_total_ns as f64) / 1.0e6;
        let n = f64::from(self.decode_steps.max(1));
        tracing::info!(
            target: "decode_profile",
            arch,
            n_steps = self.decode_steps,
            prefill_ms,
            forward_total_ms = forward_ms,
            eval_total_ms = eval_ms,
            step_total_ms = step_ms,
            forward_per_step_ms = forward_ms / n,
            eval_per_step_ms = eval_ms / n,
            "decode_profile"
        );
    }
}

#[cfg(test)]
#[path = "decode_loop_tests.rs"]
mod tests;
