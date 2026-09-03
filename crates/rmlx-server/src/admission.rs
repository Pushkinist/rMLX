//! In-process adaptive admission controller (SLA-driven 503 + adaptive max_queue_depth).
//!
//! Ship behind `--adaptive-admission` (default OFF). When OFF, this module is
//! fully inert — the existing open-loop FIFO semaphore in `engine::admit_request`
//! is the only admission path and behaviour is byte-identical to without the controller.
//!
//! ## Architecture
//!
//! Port of the **decision shape** of Dynamo's SLA-Based Planner, adapted for
//! a single in-process Mac/Metal engine (no K8s, no replica counts):
//!
//! - [`StepMetrics`] — per-request completion telemetry (queue, prompt tokens, KV bytes).
//! - [`Regressor`] — sliding-window 2D OLS: `(prompt_tokens, kv_bytes) → step_ms`.
//!   Manual least-squares, no external crate.
//! - [`AdmissionController`] — controller state: SLA targets, adaptive `max_queue_depth`
//!   (raised/lowered by the 5-s tick), anticipatory 503 at admit-time.
//! - [`ControllerHandle`] — cheaply clonable `Arc` handle exposed to the route layer.
//!
//! ## Decision-reason enum
//!
//! [`DecisionReason`] mirrors the Dynamo `planner_metrics.py:8-30` load-loop
//! vocabulary. Each admission decision is recorded as a `tracing` field and
//! (on controller ticks) as a DB `events.op` value via [`EventRecorder`].
//!
//! ## Deadband (anti-oscillation)
//!
//! Queue-depth is raised only when `est_itl < itl_target × sensitivity` (default
//! 0.80 — 80 %). Lowered only when `est_itl > itl_target` for `HOLD_TICKS` (3)
//! consecutive ticks. This prevents 1→0→1 oscillation (Dynamo sensitivity /
//! consolidation logic, simplified for N=1).
//!
//! ## References
//!
//! - Local research notes — port verdict and algorithm map.
//! - Dynamo `components/src/dynamo/planner/core/load_scaling.py:91` — deadband.
//! - Dynamo `components/src/dynamo/planner/monitoring/planner_metrics.py:8-30` — reasons.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rmlx_metrics::events::{EventRecorder, Measurement};
use serde_json;

// ── Constants ────────────────────────────────────────────────────────────────

/// Default step-target SLA value in milliseconds (Dynamo default: 500 ms).
///
/// M2: this field is the end-to-end admission→final-token wall-clock target,
/// not TTFT per se. The anticipatory-503 gate fires when the OLS prediction
/// exceeds `TTFT_REJECT_MULT × step_target_ms` (2×). Renamed from
/// `DEFAULT_TTFT_TARGET_MS`; the CLI flag `--ttft-target-ms` is kept as a
/// hidden alias for backward compatibility.
pub const DEFAULT_STEP_TARGET_MS: u64 = 500;

/// Default ITL SLA target in milliseconds (Dynamo default: 50 ms).
pub const DEFAULT_ITL_TARGET_MS: u64 = 50;
/// Sensitivity deadband for queue-depth raise (80 %). Raise fires only when
/// `est_itl < itl_target * SENSITIVITY`.
const SENSITIVITY: f64 = 0.80;
/// Controller tick interval (5 s — same as Dynamo load-loop default).
const TICK_INTERVAL: Duration = Duration::from_secs(5);
/// Minimum window points before the regressor is considered reliable.
const MIN_REGRESSION_POINTS: usize = 4;
/// Maximum window size for the OLS sliding window.
const WINDOW_SIZE: usize = 64;
/// Number of consecutive ticks `est_itl > itl_target` required before
/// lowering `max_queue_depth` (prevents single-spike flaps).
const HOLD_TICKS: usize = 3;
/// Absolute minimum queue depth the controller will ever set (1 = always
/// admit at least one request, otherwise the server would deadlock).
const MIN_QUEUE_DEPTH: usize = 1;
/// Absolute maximum queue depth the controller will ever set adaptively.
const MAX_QUEUE_DEPTH_CEIL: usize = 256;
/// Anticipatory-503 threshold multiplier. Reject if `est_ttft > TTFT_REJECT_MULT × ttft_target`.
const TTFT_REJECT_MULT: f64 = 2.0;

// ── Adaptive prefill-chunk constants ─────────────────────────────────────────

/// Prefill-chunk bounds for adaptive adjustment.
///
/// Read from `rmlx_models::prefill_chunk` rather than restated, because they
/// are the bounds `set_prefill_chunk` clamps to: a local literal that drifted
/// low would make the controller compute raises it cannot reach, and one that
/// drifted high would have it report a chunk the setter silently reduced.
const ADAPTIVE_CHUNK_MIN: usize = rmlx_models::prefill_chunk::PREFILL_CHUNK_MIN;
const ADAPTIVE_CHUNK_MAX: usize = rmlx_models::prefill_chunk::PREFILL_CHUNK_MAX;

// ── Decision-reason enum ─────────────────────────────────────────────────────

/// Decision reason from each controller evaluation.
///
/// Mirrors Dynamo `planner_metrics.py:8-30` load-loop vocabulary, adapted for
/// the single-process in-process case (no K8s / no replica concepts).
///
/// Used as:
/// 1. A `tracing` field on every admission gate event.
/// 2. The `events.op` value written by the controller tick to the DB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive] // re-exported via lib.rs — kept #[non_exhaustive] for downstream forward-compat
pub enum DecisionReason {
    /// Controller is disabled (`--adaptive-admission` not set). This reason is
    /// never written to the events table — it means the whole module is inactive.
    Disabled,
    /// Fewer than `MIN_REGRESSION_POINTS` samples in the window; no prediction
    /// possible. Admission open; no depth adjustment.
    InsufficientData,
    /// Both TTFT and ITL estimates are within SLA. No change to queue depth.
    NoChange,
    /// ITL estimate exceeds SLA for `HOLD_TICKS` ticks; queue depth lowered.
    ScaleDown,
    /// ITL estimate comfortably below SLA deadband; queue depth raised.
    ScaleUp,
    /// TTFT estimate exceeds `TTFT_REJECT_MULT × ttft_target`; this specific
    /// admission attempt is rejected with a 503 anticipatory refusal.
    AnticipatorySlo503,
    /// Adaptive prefill chunk was raised (load below deadband).
    PrefillChunkRaise,
    /// Adaptive prefill chunk was lowered (sustained overload).
    PrefillChunkLower,
    /// Adaptive prefill chunk evaluated but held (deadband or hold-tick gate).
    PrefillChunkHold,
}

impl DecisionReason {
    /// Stable snake_case string used as `events.op` and tracing field value.
    pub fn as_str(self) -> &'static str {
        match self {
            DecisionReason::Disabled => "admission_disabled",
            DecisionReason::InsufficientData => "admission_insufficient_data",
            DecisionReason::NoChange => "admission_no_change",
            DecisionReason::ScaleDown => "admission_scale_down",
            DecisionReason::ScaleUp => "admission_scale_up",
            DecisionReason::AnticipatorySlo503 => "admission_anticipatory_503",
            DecisionReason::PrefillChunkRaise => "prefill_chunk_raise",
            DecisionReason::PrefillChunkLower => "prefill_chunk_lower",
            DecisionReason::PrefillChunkHold => "prefill_chunk_hold",
        }
    }
}

// ── Per-request completion telemetry ────────────────────────────────────────

/// Telemetry emitted by the decode thread after each request completes.
///
/// Consolidates fields collected separately in `engine.rs` / `openai.rs` into a
/// single struct matching the Dynamo `ForwardPassMetrics` shape (prompt-token
/// backlog + KV footprint → step wall time).
#[derive(Debug, Clone)]
#[non_exhaustive] // re-exported via lib.rs — kept #[non_exhaustive] for downstream forward-compat
pub struct StepMetrics {
    /// Number of prompt tokens admitted in this request.
    pub prompt_tokens: u64,
    /// Total KV-cache bytes allocated at request completion.
    pub decode_kv_bytes: u64,
    /// Queue depth at the time of admission (inclusive).
    pub queue_depth: u64,
    /// Time the request spent waiting in the FIFO queue (ms).
    pub queue_wait_ms: u64,
    /// Wall-clock time from admission to final token (ms).
    /// Serves as `step_ms` — the regression target.
    pub step_ms: u64,
}

// ── 2D OLS sliding-window regressor ─────────────────────────────────────────

/// Sliding-window 2D ordinary least-squares regressor.
///
/// Fits: `step_ms ≈ a₀ + a₁ × prompt_tokens + a₂ × kv_bytes`
/// using the last `WINDOW_SIZE` `(prompt_tokens, kv_bytes, step_ms)` tuples.
///
/// Manual closed-form 2D OLS (~80 LOC). No external crate dependency.
///
/// # Usage
/// ```rust,ignore
/// let mut r = Regressor::new();
/// r.push(128, 0, 45.0);
/// if let Some(est) = r.predict_step_ms(256, 0) { … }
/// ```
#[derive(Debug)]
pub struct Regressor {
    window: VecDeque<(f64, f64, f64)>, // (prompt_tokens, kv_bytes, step_ms)
}

impl Regressor {
    /// Create a new empty regressor.
    pub fn new() -> Self {
        Self {
            window: VecDeque::with_capacity(WINDOW_SIZE),
        }
    }

    /// Add a new observation. Evicts the oldest entry when `WINDOW_SIZE` is exceeded.
    pub fn push(&mut self, prompt_tokens: u64, kv_bytes: u64, step_ms: f64) {
        if self.window.len() >= WINDOW_SIZE {
            self.window.pop_front();
        }
        self.window
            .push_back((prompt_tokens as f64, kv_bytes as f64, step_ms));
    }

    /// Number of observations currently in the window.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Returns `true` when the window has no observations.
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Returns `true` when there are fewer than `MIN_REGRESSION_POINTS` observations.
    pub fn is_insufficient(&self) -> bool {
        self.window.len() < MIN_REGRESSION_POINTS
    }

    /// Predict `step_ms` for a hypothetical new request with `prompt_tokens`
    /// prompt tokens and `kv_bytes` KV allocation.
    ///
    /// Returns `None` when there are fewer than `MIN_REGRESSION_POINTS`
    /// observations (regression is unreliable).
    ///
    /// # Algorithm
    ///
    /// Closed-form OLS for `y = β₀ + β₁x₁ + β₂x₂`:
    ///
    /// Build the design matrix X (n × 3, first column = 1), then:
    ///   β = (XᵀX)⁻¹ Xᵀy
    ///
    /// For n points with features (x1_i, x2_i) and targets y_i:
    ///   Xᵀy = [Σy, Σx1·y, Σx2·y]
    ///   XᵀX = [[n, Σx1, Σx2], [Σx1, Σx1², Σx1x2], [Σx2, Σx1x2, Σx2²]]
    ///
    /// Invert the 3×3 matrix analytically (Cramer's rule). If the determinant
    /// is near-zero (collinear / constant inputs) fall back to the mean.
    #[allow(
        clippy::suboptimal_flops,
        reason = "OLS coefficient math — fused MAD/MSUB unnecessary at this call frequency"
    )]
    pub fn predict_step_ms(&self, prompt_tokens: u64, kv_bytes: u64) -> Option<f64> {
        let n = self.window.len();
        if n < MIN_REGRESSION_POINTS {
            return None;
        }

        // Accumulate sums.
        let (mut s1, mut s2, mut sy) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut s11, mut s12, mut s22) = (0.0_f64, 0.0_f64, 0.0_f64);
        let (mut s1y, mut s2y) = (0.0_f64, 0.0_f64);
        let fn_val = n as f64;

        for &(x1, x2, y) in &self.window {
            s1 += x1;
            s2 += x2;
            sy += y;
            s11 += x1 * x1;
            s12 += x1 * x2;
            s22 += x2 * x2;
            s1y += x1 * y;
            s2y += x2 * y;
        }

        // XᵀX rows:  [n, s1, s2]
        //             [s1, s11, s12]
        //             [s2, s12, s22]
        // Xᵀy = [sy, s1y, s2y]

        // Determinant of XᵀX (3×3 expansion along first row).
        let det = fn_val * (s11 * s22 - s12 * s12) - s1 * (s1 * s22 - s12 * s2)
            + s2 * (s1 * s12 - s11 * s2);

        if det.abs() < 1e-12 {
            // Near-singular matrix: the two features may be collinear or one may
            // be entirely zero (e.g. all kv_bytes == 0 in early warm-up).
            // Try simple 1D OLS on prompt_tokens alone before falling back to mean.
            let denom1d = fn_val * s11 - s1 * s1;
            if denom1d.abs() > 1e-12 {
                let b1 = (fn_val * s1y - s1 * sy) / denom1d;
                let b0 = (sy - b1 * s1) / fn_val;
                let x1 = prompt_tokens as f64;
                return Some((b0 + b1 * x1).max(0.0));
            }
            // Constant / zero-variance in both features: fall back to mean.
            return Some(sy / fn_val);
        }

        // Cramer's rule: β = adj(XᵀX)ᵀ · Xᵀy / det
        // Cofactor matrix of XᵀX (symmetric, so adjugate = cofactor transposed = cofactor).
        // L4: c10==c01, c20==c02, c21==c12 by symmetry — use the canonical names directly.
        let c00 = s11 * s22 - s12 * s12;
        let c01 = -(s1 * s22 - s12 * s2);
        let c02 = s1 * s12 - s11 * s2;
        let c11 = fn_val * s22 - s2 * s2;
        let c12 = -(fn_val * s12 - s1 * s2);
        let c22 = fn_val * s11 - s1 * s1;

        let b0 = (c00 * sy + c01 * s1y + c02 * s2y) / det;
        let b1 = (c01 * sy + c11 * s1y + c12 * s2y) / det;
        let b2 = (c02 * sy + c12 * s1y + c22 * s2y) / det;

        let x1 = prompt_tokens as f64;
        let x2 = kv_bytes as f64;
        // Clamp: predictions must be non-negative (latency cannot be negative).
        Some((b0 + b1 * x1 + b2 * x2).max(0.0))
    }
}

impl Default for Regressor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Controller config ────────────────────────────────────────────────────────

/// Startup configuration for the adaptive controller.
#[derive(Debug, Clone)]
#[non_exhaustive]
// re-exported via lib.rs — kept #[non_exhaustive] for downstream forward-compat
pub struct ControllerConfig {
    /// End-to-end step SLA target in ms (`--step-target-ms`, default 500).
    ///
    /// M2: this is the admission→final-token wall-clock target. The anticipatory
    /// 503 fires when the OLS prediction exceeds `2 × step_target_ms`. Renamed
    /// from `ttft_target_ms`; the CLI flag `--ttft-target-ms` is a hidden alias.
    pub step_target_ms: u64,
    /// ITL SLA target in ms (`--itl-target-ms`, default 50).
    pub itl_target_ms: u64,
    /// Initial (and minimum) queue depth. Equals the static `--max-queue-depth`
    /// passed at startup. The controller never goes below `MIN_QUEUE_DEPTH` (1).
    pub initial_queue_depth: usize,
    /// Module-style arch key of the loaded model (`"gemma4"`, `"qwen3_5_moe"`, …),
    /// used to resolve the arch's prefill-chunk default when no runtime override
    /// is active. Empty string → conservative FALLBACK chunk.
    pub arch: String,
    /// When `true`, the controller also adjusts the process-wide prefill
    /// chunk size via `rmlx_models::prefill_chunk::set_prefill_chunk`.
    ///
    /// OFF by default (`--adaptive-prefill-chunk` CLI flag). The adjustment
    /// uses the same deadband shape as queue-depth but operates independently:
    /// raise when load is comfortably below ITL deadband; lower after
    /// `HOLD_TICKS` consecutive overload ticks.
    pub adaptive_prefill_chunk: bool,
}

impl ControllerConfig {
    /// Construct a new `ControllerConfig` with `adaptive_prefill_chunk = false`.
    ///
    /// `arch` is the module-style key of the loaded model (see [`ControllerConfig::arch`]).
    pub fn new(
        step_target_ms: u64,
        itl_target_ms: u64,
        initial_queue_depth: usize,
        arch: String,
    ) -> Self {
        Self {
            step_target_ms,
            itl_target_ms,
            initial_queue_depth,
            arch,
            adaptive_prefill_chunk: false,
        }
    }

    /// Enable or disable adaptive prefill-chunk sizing. Returns `self`.
    #[must_use]
    pub fn with_adaptive_prefill_chunk(mut self, enabled: bool) -> Self {
        self.adaptive_prefill_chunk = enabled;
        self
    }
}

// ── Internal mutable state ───────────────────────────────────────────────────

// N1: EventRecorder implements Debug (manual impl in rmlx_metrics::events),
// so all fields implement Debug — derive is sufficient.
#[derive(Debug)]
struct Inner {
    config: ControllerConfig,
    regressor: Regressor,
    /// Current adaptive `max_queue_depth`. Starts at `config.initial_queue_depth`.
    current_depth: usize,
    /// Consecutive ticks where `est_itl > itl_target` (for HOLD_TICKS gate on queue depth).
    overload_ticks: usize,
    /// Consecutive ticks where `est_itl > itl_target` for the prefill-chunk lower gate.
    /// Separate from `overload_ticks` so prefill-chunk and queue-depth adjustments
    /// are independent and do not reset each other's hold counters.
    prefill_chunk_overload_ticks: usize,
    /// Wall-clock of the last controller tick.
    last_tick: Instant,
    /// EventRecorder for writing tick diagnostics to the `events` table.
    /// `None` in unit-test paths that do not wire the DB.
    recorder: Option<Arc<EventRecorder>>,
}

// ── Public handle ────────────────────────────────────────────────────────────

/// Cheaply clonable handle to the admission controller.
///
/// One instance lives on `AppState`; route handlers clone it for each request.
/// `None` → adaptive mode is OFF; open-loop path is unchanged.
#[derive(Clone, Debug)]
pub struct ControllerHandle {
    inner: Arc<Mutex<Inner>>,
    /// Atomic mirror of `Inner::current_depth` for lock-free reads in
    /// the hot admission path. Written only by the tick loop (under lock).
    pub current_depth: Arc<AtomicUsize>,
}

impl ControllerHandle {
    /// Construct a new enabled controller handle.
    pub fn new(config: ControllerConfig, recorder: Option<Arc<EventRecorder>>) -> Self {
        let init = config
            .initial_queue_depth
            .clamp(MIN_QUEUE_DEPTH, MAX_QUEUE_DEPTH_CEIL);
        let depth_atom = Arc::new(AtomicUsize::new(init));
        let inner = Inner {
            current_depth: init,
            config,
            regressor: Regressor::new(),
            overload_ticks: 0,
            prefill_chunk_overload_ticks: 0,
            last_tick: Instant::now(),
            recorder,
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
            current_depth: depth_atom,
        }
    }

    /// Record a completed request's telemetry into the regressor window.
    ///
    /// Called from the blocking decode thread (after all tokens produced).
    /// Lock contention is minimal: only the tick loop also holds this lock.
    pub fn record_step(&self, m: &StepMetrics) {
        let mut g = self.inner.lock();
        g.regressor
            .push(m.prompt_tokens, m.decode_kv_bytes, m.step_ms as f64);
    }

    /// Admission gate: called from `admit_request` when the controller is enabled.
    ///
    /// Returns `Some(reason)` when the request should be rejected with a 503
    /// anticipatory refusal (`reason == AnticipatorySlo503`). Returns `None`
    /// to proceed with normal admission.
    ///
    /// `prompt_tokens` — token count of the incoming request (estimated from
    /// `gen_req.prompt_tokens.len()` at the call site, before the GPU permit
    /// is acquired).
    /// `current_kv_bytes` — current KV allocation from the resident generator.
    ///   Pass 0 when not available (conservative: makes the estimate less accurate
    ///   but never over-rejects).
    pub fn check_admission(
        &self,
        prompt_tokens: u64,
        current_kv_bytes: u64,
    ) -> Option<DecisionReason> {
        let g = self.inner.lock();
        if g.regressor.is_insufficient() {
            tracing::debug!(
                window = g.regressor.len(),
                min_required = MIN_REGRESSION_POINTS,
                "admission_ctrl: insufficient_data — pass-through"
            );
            return None;
        }
        let est_step_ms = g
            .regressor
            .predict_step_ms(prompt_tokens, current_kv_bytes)?;
        // M2: compare end-to-end step prediction against 2× step_target_ms.
        // Renamed from ttft_threshold; semantics are "est wall-clock step exceeds
        // 2× step-target", not TTFT per se.
        let step_threshold = g.config.step_target_ms as f64 * TTFT_REJECT_MULT;
        if est_step_ms > step_threshold {
            tracing::warn!(
                est_step_ms,
                step_threshold,
                prompt_tokens,
                current_kv_bytes,
                reason = DecisionReason::AnticipatorySlo503.as_str(),
                "admission_ctrl: anticipatory 503 — est end-to-end step exceeds 2× step target"
            );
            return Some(DecisionReason::AnticipatorySlo503);
        }
        None
    }

    /// Drive one controller tick. Should be called from the background task
    /// spawned by [`spawn_controller_task`].
    ///
    /// Evaluates the regression at (0 new prompt tokens, current KV footprint
    /// ≈ mean observed) and adjusts `current_depth` per the deadband rules:
    ///
    /// - `est_itl > itl_target` for `HOLD_TICKS` consecutive ticks → scale down.
    /// - `est_itl < itl_target × SENSITIVITY` → scale up.
    /// - Otherwise → no change.
    ///
    /// The "est_itl" proxy is `predict_step_ms(0, mean_kv)` divided by the mean
    /// decode token count (1 token per step ≈ ITL). For a single-step engine the
    /// per-step wall time is a direct ITL proxy.
    #[allow(
        clippy::cognitive_complexity,
        reason = "SLA planner deadband logic — three branches × hold-tick gate is inherent"
    )]
    pub fn tick(&self) {
        let mut g = self.inner.lock();
        // Throttle: only run once per TICK_INTERVAL.
        if g.last_tick.elapsed() < TICK_INTERVAL {
            return;
        }
        g.last_tick = Instant::now();

        if g.regressor.is_insufficient() {
            tracing::debug!(
                window = g.regressor.len(),
                "admission_ctrl tick: insufficient_data — no adjustment"
            );
            let reason = DecisionReason::InsufficientData;
            // H1: snapshot needed data, drop guard before writing to DB.
            let rec_opt = g.recorder.clone();
            let depth_snap = g.current_depth;
            let window_len = g.regressor.len();
            drop(g);
            write_tick_event_unlocked(rec_opt, depth_snap, window_len, reason, f64::NAN, f64::NAN);
            return;
        }

        // Use mean observed features as the "steady-state" prediction point.
        let (mean_prompt, mean_kv, mean_step) = window_means(&g.regressor.window);
        // est_itl: the predicted step_ms at the mean operating point, treated as
        // a per-token ITL proxy (single-token-per-step engine).
        let est_itl = g
            .regressor
            .predict_step_ms(mean_prompt as u64, mean_kv as u64)
            .unwrap_or(mean_step);
        let itl_target = g.config.itl_target_ms as f64;
        let depth_before = g.current_depth;

        let reason = if est_itl > itl_target {
            g.overload_ticks += 1;
            if g.overload_ticks >= HOLD_TICKS && g.current_depth > MIN_QUEUE_DEPTH {
                let new_depth = (g.current_depth.saturating_sub(1)).max(MIN_QUEUE_DEPTH);
                g.current_depth = new_depth;
                g.overload_ticks = 0;
                tracing::info!(
                    est_itl,
                    itl_target,
                    depth_before,
                    new_depth,
                    reason = DecisionReason::ScaleDown.as_str(),
                    "admission_ctrl tick: scale_down"
                );
                DecisionReason::ScaleDown
            } else {
                tracing::debug!(
                    est_itl,
                    itl_target,
                    overload_ticks = g.overload_ticks,
                    hold_required = HOLD_TICKS,
                    "admission_ctrl tick: no_change (overload, hold gate)"
                );
                DecisionReason::NoChange
            }
        } else if est_itl < itl_target * SENSITIVITY && g.current_depth < MAX_QUEUE_DEPTH_CEIL {
            g.overload_ticks = 0;
            let new_depth = (g.current_depth + 1).min(MAX_QUEUE_DEPTH_CEIL);
            g.current_depth = new_depth;
            tracing::info!(
                est_itl,
                itl_target,
                depth_before,
                new_depth,
                reason = DecisionReason::ScaleUp.as_str(),
                "admission_ctrl tick: scale_up"
            );
            DecisionReason::ScaleUp
        } else {
            g.overload_ticks = 0;
            tracing::debug!(est_itl, itl_target, "admission_ctrl tick: no_change");
            DecisionReason::NoChange
        };

        // Publish the new depth to the atomic (hot-path lock-free reads).
        self.current_depth.store(g.current_depth, Ordering::Release);

        // H1: snapshot needed data, drop guard before writing to DB.
        let rec_opt = g.recorder.clone();
        let depth_snap = g.current_depth;
        let window_len = g.regressor.len();
        let adaptive_prefill = g.config.adaptive_prefill_chunk;
        drop(g);
        write_tick_event_unlocked(rec_opt, depth_snap, window_len, reason, est_itl, itl_target);

        // If adaptive prefill chunk is enabled, apply the same deadband
        // shape to the process-wide prefill chunk setting.
        if adaptive_prefill {
            self.tick_prefill_chunk(est_itl, itl_target);
        }
    }

    /// Adjust the process-wide prefill chunk if `adaptive_prefill_chunk` is enabled.
    ///
    /// Same deadband shape as queue-depth: raise when `est_itl < itl_target * SENSITIVITY`,
    /// lower after `HOLD_TICKS` consecutive overload ticks. The current chunk value is
    /// read from the `rmlx_models::prefill_chunk::runtime_override` global.
    fn tick_prefill_chunk(&self, est_itl: f64, itl_target: f64) {
        let mut g = self.inner.lock();
        let current = rmlx_models::prefill_chunk::runtime_override()
            .unwrap_or_else(|| rmlx_models::prefill_chunk::prefill_chunk_for(&g.config.arch));

        let reason = if est_itl > itl_target {
            g.prefill_chunk_overload_ticks += 1;
            if g.prefill_chunk_overload_ticks >= HOLD_TICKS && current > ADAPTIVE_CHUNK_MIN {
                let new_chunk = (current / 2).max(ADAPTIVE_CHUNK_MIN);
                g.prefill_chunk_overload_ticks = 0;
                drop(g);
                rmlx_models::prefill_chunk::set_prefill_chunk(new_chunk);
                tracing::info!(
                    est_itl,
                    itl_target,
                    old_chunk = current,
                    new_chunk,
                    reason = DecisionReason::PrefillChunkLower.as_str(),
                    "admission_ctrl: prefill_chunk lowered"
                );
                DecisionReason::PrefillChunkLower
            } else {
                drop(g);
                DecisionReason::PrefillChunkHold
            }
        } else if est_itl < itl_target * SENSITIVITY && current < ADAPTIVE_CHUNK_MAX {
            g.prefill_chunk_overload_ticks = 0;
            let new_chunk = ((current * 3) / 2).min(ADAPTIVE_CHUNK_MAX);
            drop(g);
            rmlx_models::prefill_chunk::set_prefill_chunk(new_chunk);
            tracing::info!(
                est_itl,
                itl_target,
                old_chunk = current,
                new_chunk,
                reason = DecisionReason::PrefillChunkRaise.as_str(),
                "admission_ctrl: prefill_chunk raised"
            );
            DecisionReason::PrefillChunkRaise
        } else {
            g.prefill_chunk_overload_ticks = 0;
            drop(g);
            DecisionReason::PrefillChunkHold
        };

        tracing::debug!(
            reason = reason.as_str(),
            current_chunk = current,
            "admission_ctrl: prefill_chunk tick"
        );
    }

    /// Force a controller tick, bypassing the `TICK_INTERVAL` guard.
    ///
    /// L2: test-only hook. Resets `last_tick` to the epoch and calls `tick()`
    /// so tests do not need the fragile `Instant::now().checked_sub(...)` trick
    /// (which inverts intent on systems where `Instant::now()` is very small).
    #[cfg(test)]
    pub(crate) fn tick_force(&self) {
        {
            let mut g = self.inner.lock();
            g.last_tick = Instant::now()
                .checked_sub(TICK_INTERVAL + Duration::from_secs(1))
                .unwrap_or_else(Instant::now);
        }
        self.tick();
    }
}

// ── Background task handle ────────────────────────────────────────────────────

/// RAII wrapper around the admission controller's background tokio task.
///
/// Aborting the task on `Drop` ensures the tick loop exits promptly when the
/// server shuts down (runtime teardown drops `AppState`, which drops this
/// handle). The abort causes the next `interval.tick().await` to return an
/// error, at which point the task exits cleanly within at most 1 s.
///
/// Stored as `AppState::admission_handle: Option<AdmissionHandle>`.
pub struct AdmissionHandle {
    join: tokio::task::JoinHandle<()>,
}

impl Drop for AdmissionHandle {
    fn drop(&mut self) {
        self.join.abort();
    }
}

impl std::fmt::Debug for AdmissionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmissionHandle")
            .field("is_finished", &self.join.is_finished())
            .finish()
    }
}

// ── Background task launcher ─────────────────────────────────────────────────

/// Spawn a background tokio task that drives [`ControllerHandle::tick`] every
/// second (fine-grained; the tick itself throttles to `TICK_INTERVAL`).
///
/// The returned [`AdmissionHandle`] should be stored on `AppState`. It aborts
/// the tick task when dropped (graceful shutdown path — tokio abort cancels the
/// next `interval.tick().await` within at most 1 s).
pub fn spawn_controller_task(handle: ControllerHandle) -> AdmissionHandle {
    let join = tokio::spawn(async move {
        tracing::info!(
            tick_interval_ms = TICK_INTERVAL.as_millis() as u64,
            "admission controller tick task started"
        );
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            handle.tick();
        }
    });
    AdmissionHandle { join }
}

// ── Private helpers ──────────────────────────────────────────────────────────

fn window_means(window: &VecDeque<(f64, f64, f64)>) -> (f64, f64, f64) {
    if window.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = window.len() as f64;
    let (mut s1, mut s2, mut sy) = (0.0_f64, 0.0_f64, 0.0_f64);
    for &(x1, x2, y) in window {
        s1 += x1;
        s2 += x2;
        sy += y;
    }
    (s1 / n, s2 / n, sy / n)
}

/// Write a tick event to the DB without holding `Inner`'s Mutex.
///
/// H1: caller must snapshot the needed fields and drop the `Inner` guard before
/// calling this. `EventRecorder::record` acquires its own mutex + does a sync
/// SQLite INSERT; executing that while holding `Inner`'s lock would stall the
/// hot admission path (`check_admission`, `record_step`) for the full DB round-trip.
fn write_tick_event_unlocked(
    rec_opt: Option<Arc<EventRecorder>>,
    depth: usize,
    window_len: usize,
    reason: DecisionReason,
    est_itl: f64,
    itl_target: f64,
) {
    if let Some(rec) = rec_opt {
        // N2: sanitise non-finite floats — JSON does not allow NaN/Infinity.
        let safe_itl = if est_itl.is_finite() { est_itl } else { 0.0 };
        let safe_target = if itl_target.is_finite() {
            itl_target
        } else {
            0.0
        };
        let notes = serde_json::json!({
            "depth": depth,
            "est_itl_ms": safe_itl,
            "itl_target_ms": safe_target,
            "window": window_len,
            "reason": reason.as_str(),
        })
        .to_string();
        if let Err(e) = rec.record(&Measurement {
            model_path: "server",
            quant_mode: "n/a",
            stage: "admission_ctrl",
            op: reason.as_str(),
            value_unit: "ms",
            value: safe_itl,
            notes: &notes,
        }) {
            tracing::warn!(error = %e, "admission_ctrl: events-table write failed");
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "admission_tests.rs"]
mod tests;
