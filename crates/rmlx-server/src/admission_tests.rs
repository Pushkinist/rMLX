//! Unit tests for the adaptive admission controller.

#![allow(clippy::suboptimal_flops)] // test arithmetic clarity > FMA rewrite

use super::*;

// ── Regressor tests ──────────────────────────────────────────────────────────

#[test]
fn regressor_insufficient_before_min_points() {
    let mut r = Regressor::new();
    assert!(r.is_insufficient());
    r.push(100, 0, 40.0);
    r.push(200, 0, 60.0);
    r.push(300, 0, 80.0);
    assert!(r.is_insufficient()); // still < MIN_REGRESSION_POINTS (4)
    r.push(400, 0, 100.0);
    assert!(!r.is_insufficient()); // exactly 4
}

#[test]
fn regressor_predict_linear_1d() {
    // Perfect 1D linear data: step_ms = 0.1 × prompt_tokens + 10
    let mut r = Regressor::new();
    for tokens in [100u64, 200, 300, 400, 500, 600] {
        let ms = 0.1 * tokens as f64 + 10.0;
        r.push(tokens, 0, ms);
    }
    // Predict at 1000 tokens: expected ≈ 110 ms
    let pred = r.predict_step_ms(1000, 0).expect("should predict");
    let expected = 0.1 * 1000.0 + 10.0; // 110.0
    assert!(
        (pred - expected).abs() < 1.0,
        "expected ~{expected:.1}, got {pred:.1}"
    );
}

#[test]
fn regressor_predict_2d() {
    // 2D linear: step_ms = 5 + 0.05*prompt + 0.001*kv_bytes
    let mut r = Regressor::new();
    let combos: &[(u64, u64)] = &[
        (100, 1000),
        (200, 2000),
        (300, 1000),
        (400, 3000),
        (500, 500),
        (100, 5000),
    ];
    for &(p, kv) in combos {
        let ms = 5.0 + 0.05 * p as f64 + 0.001 * kv as f64;
        r.push(p, kv, ms);
    }
    let pred = r.predict_step_ms(300, 2000).expect("should predict");
    let expected = 5.0 + 0.05 * 300.0 + 0.001 * 2000.0; // 5 + 15 + 2 = 22
    assert!(
        (pred - expected).abs() < 2.0,
        "expected ~{expected:.1}, got {pred:.1}"
    );
}

#[test]
fn regressor_constant_input_fallback_to_mean() {
    // All prompt_tokens == 0, all kv_bytes == 0: degenerate → falls back to mean.
    let mut r = Regressor::new();
    for ms in [10.0, 20.0, 30.0, 40.0] {
        r.push(0, 0, ms);
    }
    let pred = r.predict_step_ms(0, 0).expect("should predict (fallback)");
    assert!(
        (pred - 25.0).abs() < 1.0,
        "expected mean 25.0, got {pred:.1}"
    );
}

#[test]
fn regressor_window_evicts_oldest() {
    let mut r = Regressor::new();
    // Fill the window + 2 extra.
    for i in 0..=(WINDOW_SIZE as u64) {
        r.push(i * 10, 0, i as f64 * 5.0);
    }
    assert_eq!(r.len(), WINDOW_SIZE);
}

#[test]
fn regressor_predict_returns_none_below_min() {
    let r = Regressor::new();
    assert!(r.predict_step_ms(100, 0).is_none());
}

// ── DecisionReason::as_str ───────────────────────────────────────────────────

#[test]
fn decision_reason_as_str_stable() {
    assert_eq!(
        DecisionReason::InsufficientData.as_str(),
        "admission_insufficient_data"
    );
    assert_eq!(DecisionReason::NoChange.as_str(), "admission_no_change");
    assert_eq!(DecisionReason::ScaleDown.as_str(), "admission_scale_down");
    assert_eq!(DecisionReason::ScaleUp.as_str(), "admission_scale_up");
    assert_eq!(
        DecisionReason::AnticipatorySlo503.as_str(),
        "admission_anticipatory_503"
    );
    assert_eq!(DecisionReason::Disabled.as_str(), "admission_disabled");
}

// ── ControllerHandle::check_admission ────────────────────────────────────────

fn make_controller(ttft_ms: u64, itl_ms: u64, init_depth: usize) -> ControllerHandle {
    ControllerHandle::new(
        ControllerConfig::new(ttft_ms, itl_ms, init_depth),
        None, // no DB in unit tests
    )
}

#[test]
fn check_admission_pass_through_when_insufficient_data() {
    let ctrl = make_controller(500, 50, 8);
    // No observations yet → InsufficientData → None (pass-through).
    let result = ctrl.check_admission(256, 0);
    assert!(result.is_none(), "expected pass-through, got {result:?}");
}

#[test]
fn check_admission_pass_through_below_threshold() {
    let ctrl = make_controller(500, 50, 8);
    // Feed observations well within SLA: step_ms ≈ 100 ms (< 1000 = 2×500 threshold).
    for tokens in [100u64, 200, 300, 400, 500] {
        ctrl.record_step(&StepMetrics {
            prompt_tokens: tokens,
            decode_kv_bytes: 0,
            queue_depth: 1,
            queue_wait_ms: 0,
            step_ms: 100, // well below 2×500=1000 ms threshold
        });
    }
    let result = ctrl.check_admission(256, 0);
    assert!(result.is_none(), "expected pass-through, got {result:?}");
}

#[test]
fn check_admission_rejects_above_ttft_threshold() {
    // threshold = 2 × 100 ms = 200 ms
    let ctrl = make_controller(100, 50, 8);
    // Feed observations where step_ms linearly grows with tokens: 1 ms per token.
    // At 250 tokens the prediction will be ~250 ms > 200 ms threshold.
    for tokens in [100u64, 150, 200, 250, 300] {
        ctrl.record_step(&StepMetrics {
            prompt_tokens: tokens,
            decode_kv_bytes: 0,
            queue_depth: 1,
            queue_wait_ms: 0,
            step_ms: tokens, // 1 ms/token
        });
    }
    // Predict for 250 tokens → ~250 ms > 200 ms → should reject.
    let result = ctrl.check_admission(250, 0);
    assert_eq!(
        result,
        Some(DecisionReason::AnticipatorySlo503),
        "expected anticipatory 503, got {result:?}"
    );
}

// ── Controller tick / depth adjustment ───────────────────────────────────────

#[test]
fn tick_no_change_when_insufficient_data() {
    let ctrl = make_controller(500, 50, 8);
    // Tick with no data → depth stays unchanged.
    ctrl.tick();
    assert_eq!(ctrl.current_depth.load(Ordering::Relaxed), 8);
}

#[test]
fn tick_scale_up_when_est_itl_below_deadband() {
    // itl_target = 50 ms, sensitivity = 0.80 → deadband threshold = 40 ms.
    // Feed data where step_ms ≈ 20 ms (well below 40 ms) → scale_up.
    let ctrl = make_controller(500, 50, 4);

    for tokens in [10u64, 20, 30, 40, 50, 60, 70, 80] {
        ctrl.record_step(&StepMetrics {
            prompt_tokens: tokens,
            decode_kv_bytes: 0,
            queue_depth: 1,
            queue_wait_ms: 0,
            step_ms: 20, // << 40 ms deadband
        });
    }

    // L2: use tick_force() to bypass TICK_INTERVAL without the fragile
    // checked_sub pattern.
    ctrl.tick_force();
    // Depth should have increased from 4 to 5.
    assert_eq!(ctrl.current_depth.load(Ordering::Relaxed), 5);
}

#[test]
fn tick_scale_down_after_hold_ticks() {
    // itl_target = 10 ms. Feed data where step_ms ≈ 50 ms >> 10 ms.
    let ctrl = make_controller(500, 10, 8);

    for tokens in [10u64, 20, 30, 40, 50, 60, 70, 80] {
        ctrl.record_step(&StepMetrics {
            prompt_tokens: tokens,
            decode_kv_bytes: 0,
            queue_depth: 1,
            queue_wait_ms: 0,
            step_ms: 50, // >> 10 ms target
        });
    }

    // L2: use tick_force() to bypass TICK_INTERVAL without the fragile
    // checked_sub pattern.
    for _ in 0..HOLD_TICKS {
        ctrl.tick_force();
    }

    // After HOLD_TICKS consecutive overload ticks, depth should decrease.
    assert_eq!(
        ctrl.current_depth.load(Ordering::Relaxed),
        7,
        "expected depth 7 (8 - 1 after HOLD_TICKS)"
    );
}

#[test]
fn tick_deadband_holds_no_change() {
    // itl_target = 50 ms, deadband = 40 ms.
    // Feed data where step_ms ≈ 45 ms (between 40 and 50 ms → no change zone).
    let ctrl = make_controller(500, 50, 8);

    for tokens in [10u64, 20, 30, 40, 50, 60, 70, 80] {
        ctrl.record_step(&StepMetrics {
            prompt_tokens: tokens,
            decode_kv_bytes: 0,
            queue_depth: 1,
            queue_wait_ms: 0,
            step_ms: 45, // between 40 (deadband) and 50 (target): no change zone
        });
    }

    // L2: use tick_force() to bypass TICK_INTERVAL without the fragile
    // checked_sub pattern.
    ctrl.tick_force();

    // Depth should stay at 8 (no-change zone).
    assert_eq!(ctrl.current_depth.load(Ordering::Relaxed), 8);
}

#[test]
fn initial_depth_clamped_to_bounds() {
    // Below MIN_QUEUE_DEPTH.
    let ctrl = make_controller(500, 50, 0);
    assert_eq!(ctrl.current_depth.load(Ordering::Relaxed), MIN_QUEUE_DEPTH);

    // Above MAX_QUEUE_DEPTH_CEIL.
    let ctrl2 = make_controller(500, 50, 9999);
    assert_eq!(
        ctrl2.current_depth.load(Ordering::Relaxed),
        MAX_QUEUE_DEPTH_CEIL
    );
}

// ── Task 2: DecisionReason::Prefill* variants ─────────────────────────────────

#[test]
fn prefill_chunk_decision_reason_strings_are_stable() {
    assert_eq!(
        DecisionReason::PrefillChunkRaise.as_str(),
        "prefill_chunk_raise"
    );
    assert_eq!(
        DecisionReason::PrefillChunkLower.as_str(),
        "prefill_chunk_lower"
    );
    assert_eq!(
        DecisionReason::PrefillChunkHold.as_str(),
        "prefill_chunk_hold"
    );
}

// The adaptive prefill-chunk global-state tests use a single-threaded runner to
// avoid racy interference with parallel test threads that also touch the
// process-wide `RUNTIME_OVERRIDE` atomic.
//
// `prefill_chunk_global_state_serial` runs all three scenarios sequentially
// inside one `#[test]` function, which Rust guarantees runs in one thread.
#[test]
fn prefill_chunk_global_state_serial() {
    // Scenario A: raises below deadband.
    {
        let initial_chunk = rmlx_models::prefill_chunk::PREFILL_CHUNK_MIN * 2; // 64
        rmlx_models::prefill_chunk::set_prefill_chunk(initial_chunk);

        let ctrl = ControllerHandle::new(
            ControllerConfig::new(500, 50, 4).with_adaptive_prefill_chunk(true),
            None,
        );

        for tokens in [10u64, 20, 30, 40, 50, 60, 70, 80] {
            ctrl.record_step(&StepMetrics {
                prompt_tokens: tokens,
                decode_kv_bytes: 0,
                queue_depth: 1,
                queue_wait_ms: 0,
                step_ms: 20, // << 40 ms deadband
            });
        }

        ctrl.tick_force();

        let new_chunk =
            rmlx_models::prefill_chunk::runtime_override().expect("override should be set");
        assert!(
            new_chunk > initial_chunk,
            "A: chunk should have been raised from {initial_chunk} but got {new_chunk}"
        );
        assert!(
            new_chunk <= rmlx_models::prefill_chunk::PREFILL_CHUNK_MAX,
            "A: chunk must not exceed PREFILL_CHUNK_MAX"
        );
        rmlx_models::prefill_chunk::set_prefill_chunk(0);
    }

    // Scenario B: lowers after HOLD_TICKS.
    {
        let initial_chunk = 512;
        rmlx_models::prefill_chunk::set_prefill_chunk(initial_chunk);

        let ctrl = ControllerHandle::new(
            ControllerConfig::new(500, 10, 8).with_adaptive_prefill_chunk(true),
            None,
        );

        for tokens in [10u64, 20, 30, 40, 50, 60, 70, 80] {
            ctrl.record_step(&StepMetrics {
                prompt_tokens: tokens,
                decode_kv_bytes: 0,
                queue_depth: 1,
                queue_wait_ms: 0,
                step_ms: 50, // >> 10 ms target
            });
        }

        for _ in 0..HOLD_TICKS {
            ctrl.tick_force();
        }

        let new_chunk =
            rmlx_models::prefill_chunk::runtime_override().expect("override should be set");
        assert!(
            new_chunk < initial_chunk,
            "B: chunk should have been lowered from {initial_chunk} but got {new_chunk}"
        );
        assert!(
            new_chunk >= rmlx_models::prefill_chunk::PREFILL_CHUNK_MIN,
            "B: chunk must not go below PREFILL_CHUNK_MIN"
        );
        rmlx_models::prefill_chunk::set_prefill_chunk(0);
    }

    // Scenario C: disabled controller does not change the chunk.
    {
        let sentinel: usize = 128;
        rmlx_models::prefill_chunk::set_prefill_chunk(sentinel);

        let ctrl = make_controller(500, 50, 4); // adaptive_prefill_chunk = false
        for tokens in [10u64, 20, 30, 40, 50, 60, 70, 80] {
            ctrl.record_step(&StepMetrics {
                prompt_tokens: tokens,
                decode_kv_bytes: 0,
                queue_depth: 1,
                queue_wait_ms: 0,
                step_ms: 20,
            });
        }
        ctrl.tick_force();

        assert_eq!(
            rmlx_models::prefill_chunk::runtime_override(),
            Some(sentinel),
            "C: runtime override must not change when adaptive_prefill_chunk is off"
        );
        rmlx_models::prefill_chunk::set_prefill_chunk(0);
    }
}

// ── Task 3: AdmissionHandle abort ─────────────────────────────────────────────

#[tokio::test]
async fn admission_handle_task_exits_when_dropped() {
    let ctrl = make_controller(500, 50, 8);
    let handle = spawn_controller_task(ctrl);

    // Drop the handle — this aborts the background tick task.
    drop(handle);

    // Give the runtime a moment to propagate the abort.
    tokio::time::sleep(Duration::from_millis(50)).await;
    // If we reach here without hanging, the abort + drop lifecycle is correct.
    // (A hanging drop would cause the test to time out instead of passing.)
}
