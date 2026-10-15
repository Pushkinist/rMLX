use super::*;
use futures::StreamExt;

fn sample_req() -> GenerationRequest {
    GenerationRequest {
        model_id: "test".to_owned(),
        prompt_tokens: vec![],
        max_tokens: 100,
        sampling: SamplingParams::default(),
        stop: vec![],
        stream: false,
        system: None,
        session_id: None,
        effective_prompt_cache_slots: None,
        metrics_drainer: None,
        itl_store: None,
        event_recorder: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        constraint: None,
        is_thinking_handle: None,
        thinking_budget: None,
        thinking_end_token_id: None,
        prompt_think_open: false,
        emit_tool_markers: false,
        thinking_start_token: None,
        thinking_end_token: None,
        gpu_admission: None,
        kv_quant_override: None,
        max_ctx_override: None,
        images: vec![],
        audio_b64: vec![],
        image_max_tokens: None,
    }
}

// ── FIFO admission queue tests ───────────────────────────────

use std::sync::atomic::Ordering;

/// Depth bound rejects with `QueueFull` once `gpu_pending` is at the
/// configured max — and does NOT increment the gauge on rejection.
#[tokio::test]
async fn admission_rejects_at_depth() {
    let queue = Arc::new(Semaphore::new(1));
    let pending = Arc::new(AtomicUsize::new(0));

    // Admit one (holds the only permit) — pending == 1.
    let a1 = admit_request(&queue, &pending, 2).await;
    assert!(matches!(a1, Admission::Admitted { .. }));
    assert_eq!(pending.load(Ordering::Acquire), 1);

    // Second enters the queue (waits on permit) — pending == 2.
    // Use a timeout-bounded acquire by spawning; just simulate the
    // count being at the bound via a manual fetch_add for determinism.
    pending.fetch_add(1, Ordering::AcqRel); // simulate a 2nd in-flight
    assert_eq!(pending.load(Ordering::Acquire), 2);

    // Now at depth==2, a third must be rejected WITHOUT incrementing.
    let a3 = admit_request(&queue, &pending, 2).await;
    assert!(matches!(a3, Admission::QueueFull));
    assert_eq!(
        pending.load(Ordering::Acquire),
        2,
        "rejected request must NOT increment gpu_pending"
    );
    drop(a1);
}

/// `max_queue_depth == 0` means unlimited — never rejects.
#[tokio::test]
async fn admission_unlimited_when_zero() {
    let queue = Arc::new(Semaphore::new(1));
    let pending = Arc::new(AtomicUsize::new(0));
    // Pre-load pending well past any sane bound.
    pending.store(1000, Ordering::Release);
    let a = admit_request(&queue, &pending, 0).await;
    assert!(
        matches!(a, Admission::Admitted { .. }),
        "0 = unlimited, must admit regardless of pending"
    );
    assert_eq!(pending.load(Ordering::Acquire), 1001);
    drop(a);
}

/// The RAII guard decrements `gpu_pending` and releases the permit on
/// drop — covering the success/normal completion path.
#[tokio::test]
async fn admission_guard_drop_decrements_and_releases() {
    let queue = Arc::new(Semaphore::new(1));
    let pending = Arc::new(AtomicUsize::new(0));

    let a = admit_request(&queue, &pending, 64).await;
    let guard = match a {
        Admission::Admitted { guard, depth, .. } => {
            assert_eq!(depth, 1, "first admission depth is 1");
            guard
        }
        Admission::QueueFull => panic!("should have been admitted"),
    };
    assert_eq!(pending.load(Ordering::Acquire), 1);
    assert_eq!(queue.available_permits(), 0, "permit held by guard");

    drop(guard);
    assert_eq!(
        pending.load(Ordering::Acquire),
        0,
        "guard drop must decrement gpu_pending"
    );
    assert_eq!(
        queue.available_permits(),
        1,
        "guard drop must release the permit"
    );
}

/// Error/early-drop path: a `GenerationRequest` carrying the guard that
/// is dropped WITHOUT ever reaching generation still balances the gauge
/// (the guard lives in the request; dropping the request drops it).
#[tokio::test]
async fn admission_guard_drop_via_request_balances_gauge() {
    let queue = Arc::new(Semaphore::new(1));
    let pending = Arc::new(AtomicUsize::new(0));

    let a = admit_request(&queue, &pending, 64).await;
    let guard = match a {
        Admission::Admitted { guard, .. } => guard,
        Admission::QueueFull => panic!("admitted expected"),
    };
    let mut req = sample_req();
    req.gpu_admission = Some(guard);
    assert_eq!(pending.load(Ordering::Acquire), 1);

    // Simulate an early-return path (model mismatch / empty prompt /
    // 503) where `generate()` is never spawned: the request is dropped.
    drop(req);
    assert_eq!(
        pending.load(Ordering::Acquire),
        0,
        "dropping the request (error path) must decrement gpu_pending"
    );
    assert_eq!(queue.available_permits(), 1, "permit released on drop");
}

/// FIFO fairness: two waiters resolve in strict arrival order. With a
/// 1-permit semaphore, `tokio` grants permits FIFO. The second caller
/// must not be admitted before the first releases.
#[tokio::test]
async fn admission_is_fifo() {
    let queue = Arc::new(Semaphore::new(1));
    let pending = Arc::new(AtomicUsize::new(0));
    let order = Arc::new(Mutex::new(Vec::<u32>::new()));

    // First admission takes the only permit.
    let first = admit_request(&queue, &pending, 64).await;
    let g1 = match first {
        Admission::Admitted { guard, .. } => guard,
        Admission::QueueFull => panic!("first should be admitted"),
    };

    // Spawn waiter A then waiter B (arrival order A, B).
    let (qa, pa, oa) = (Arc::clone(&queue), Arc::clone(&pending), Arc::clone(&order));
    let wa = tokio::spawn(async move {
        let a = admit_request(&qa, &pa, 64).await;
        oa.lock().push(1);
        a
    });
    // Ensure A is enqueued on the semaphore before B arrives.
    tokio::task::yield_now().await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (qb, pb, ob) = (Arc::clone(&queue), Arc::clone(&pending), Arc::clone(&order));
    let wb = tokio::spawn(async move {
        let a = admit_request(&qb, &pb, 64).await;
        ob.lock().push(2);
        a
    });
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Neither A nor B admitted yet (g1 holds the permit).
    assert!(order.lock().is_empty());

    // Release the permit: A (arrived first) must resolve before B.
    drop(g1);
    let ra = wa.await.unwrap();
    // A holds the permit now; B still blocked.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(*order.lock(), vec![1], "A must resolve first");

    // Drop A's guard → B resolves.
    if let Admission::Admitted { guard, .. } = ra {
        drop(guard);
    } else {
        panic!("A should have been admitted");
    }
    let rb = wb.await.unwrap();
    assert_eq!(
        *order.lock(),
        vec![1, 2],
        "FIFO: B resolves only after A, in arrival order"
    );
    if let Admission::Admitted { guard, .. } = rb {
        drop(guard);
    }
    assert_eq!(pending.load(Ordering::Acquire), 0, "gauge balanced");
}

#[tokio::test]
async fn not_ready_emits_exactly_one_error() {
    let gen = NotReadyGenerator;
    let mut stream = gen.generate(sample_req());

    // First item must be an error.
    let first = stream.next().await;
    match first {
        Some(Err(Error::Other(msg))) => {
            assert_eq!(msg, "generator not ready");
        }
        other => panic!("expected Err(Other(...)), got {other:?}"),
    }

    // Stream must end after the first error.
    assert!(stream.next().await.is_none(), "stream should be exhausted");
}

/// `ArchGenerator::from_snapshot` with the primary test snapshot.
///
/// Skips with a `tracing::warn!` if the snapshot directory is absent
/// (CI / machines without the model download).
#[tokio::test]
async fn gemma4_generator_loads_and_model_id_matches_basename() {
    let Some(snap) = std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
    else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping gemma4_generator_loads_and_model_id_matches_basename");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping gemma4_generator_loads_and_model_id_matches_basename"
        );
        return;
    }

    let gen = ArchGenerator::from_snapshot(
        &snap,
        &ModelLoadConfig {
            device: rmlx_mlx::Device::Cpu,
            kv_quant: None,
            max_ctx: None,
            prompt_cache_slots: 4,
            mm_cache: None,
            calibration: None,
            yarn: None,
            image_max_tokens: None,
        },
        Arc::new(Mutex::new(())),
    )
    .expect("from_snapshot must succeed when snapshot is present");

    assert_eq!(
        gen.model_id(),
        "mlx-community__gemma-4-e4b-it-mxfp8",
        "model_id must match the snapshot directory basename"
    );
}

/// Empty prompt must yield an immediate error from `ArchGenerator`.
///
/// This test uses `NotReadyGenerator` as a proxy since `ArchGenerator`
/// requires the snapshot. The empty-prompt path is tested separately
/// via the mismatch guard (both precede the spawn_blocking call).
#[tokio::test]
async fn not_ready_error_is_other_variant() {
    let gen = NotReadyGenerator;
    let req = sample_req();
    let mut stream = gen.generate(req);
    match stream.next().await {
        Some(Err(Error::Other(_))) => {}
        other => panic!("expected Err(Other), got {other:?}"),
    }
}

// ── compute_itl_stats ────────────────────────────────────────────────────

/// Verify p50/p95/mean from a known uniform 10 ms sequence.
///
/// 5 tokens → 4 intervals of exactly 10 ms each.
/// p50 = 10, p95 = 10, mean = 10.
#[test]
fn compute_itl_uniform_10ms() {
    use std::time::Duration;

    let base = Instant::now();
    let timestamps: Vec<Instant> = (0..5)
        .map(|i| base + Duration::from_millis(10 * i))
        .collect();

    let (p50, p95, p99, mean, spikes) =
        compute_itl_stats(&timestamps).expect("should compute for 5 steps");

    assert!((p50 - 10.0).abs() < 1.0, "p50 should be ~10 ms, got {p50}");
    assert!((p95 - 10.0).abs() < 1.0, "p95 should be ~10 ms, got {p95}");
    assert!((p99 - 10.0).abs() < 1.0, "p99 should be ~10 ms, got {p99}");
    assert!(
        (mean - 10.0).abs() < 1.0,
        "mean should be ~10 ms, got {mean}"
    );
    // Uniform sequence: no spikes (spike threshold = 3×p50 = 30 ms; all
    // intervals are 10 ms).
    assert_eq!(spikes, 0, "uniform sequence should have no spikes");
}

/// Verify p50/p95 diverge on a skewed sequence.
///
/// 10 intervals: 9 × 8 ms + 1 × 80 ms (10× spike at the end).
/// p50 ≈ 8, p95 ≈ 80, mean ≈ 15.2 ms.
#[test]
fn compute_itl_skewed_sequence() {
    use std::time::Duration;

    // Build timestamps from known intervals: 9 × 8 ms then 1 × 80 ms.
    let mut timestamps: Vec<Instant> = Vec::with_capacity(11);
    let base = Instant::now();
    timestamps.push(base);
    for _ in 0..9 {
        let prev = *timestamps.last().unwrap();
        timestamps.push(prev + Duration::from_millis(8));
    }
    // Spike: 80 ms after the last 8 ms step.
    let prev = *timestamps.last().unwrap();
    timestamps.push(prev + Duration::from_millis(80));

    let (p50, p95, _p99, mean, spikes) =
        compute_itl_stats(&timestamps).expect("should compute for 11 steps");

    // p50 should be in the 8 ms region (most intervals are 8 ms).
    assert!(p50 < 15.0, "p50 should be ~8 ms, got {p50}");
    // p95 should reflect the spike.
    assert!(p95 > 15.0, "p95 should be > 15 ms due to spike, got {p95}");
    // mean should be between p50 and p95.
    assert!(
        mean > p50 && mean < p95,
        "mean should lie between p50 and p95"
    );
    // 80 ms > 3×8 ms = 24 ms → the spike interval must be counted.
    assert!(
        spikes >= 1,
        "80 ms interval should count as a spike, got {spikes}"
    );
}

/// Verify p99 and spike count on a known skewed sequence.
///
/// 100 tokens → 99 intervals: 90 × 10 ms + 9 × 100 ms (9× spikes at end).
/// spike threshold = 3 × median(10 ms) = 30 ms → 9 spikes expected.
/// p99 should be in the 100 ms region.
#[test]
fn f9_compute_itl_p99_and_spikes_known_vector() {
    use std::time::Duration;

    let mut timestamps: Vec<Instant> = Vec::with_capacity(101);
    let base = Instant::now();
    timestamps.push(base);
    // 90 × 10 ms intervals.
    for _ in 0..90 {
        let prev = *timestamps.last().unwrap();
        timestamps.push(prev + Duration::from_millis(10));
    }
    // 9 × 100 ms intervals (spikes).
    for _ in 0..9 {
        let prev = *timestamps.last().unwrap();
        timestamps.push(prev + Duration::from_millis(100));
    }

    let (p50, _p95, p99, _mean, spikes) =
        compute_itl_stats(&timestamps).expect("should compute for 100 intervals");

    // p50 = 10 ms (most intervals).
    assert!(p50 < 15.0, "p50 should be ~10 ms, got {p50}");
    // p99 should be in the 100 ms region (9/99 ≈ 9% are 100 ms, p99 ≈ top).
    assert!(p99 > 50.0, "p99 should be in 100 ms region, got {p99}");
    // spike threshold = 3 × 10 = 30 ms → all 9 × 100 ms intervals are spikes.
    assert_eq!(spikes, 9, "should count exactly 9 spike intervals");
}

/// Uniform sequence has zero spikes (all intervals equal median).
#[test]
fn f9_no_spikes_uniform_sequence() {
    use std::time::Duration;
    let base = Instant::now();
    let timestamps: Vec<Instant> = (0..20)
        .map(|i| base + Duration::from_millis(10 * i))
        .collect();
    let (_p50, _p95, _p99, _mean, spikes) = compute_itl_stats(&timestamps).expect("should compute");
    assert_eq!(spikes, 0, "uniform sequence must have 0 spikes");
}

/// Fewer than 2 timestamps → None (no interval computable).
#[test]
fn compute_itl_too_few_timestamps_returns_none() {
    assert!(compute_itl_stats(&[]).is_none());
    assert!(compute_itl_stats(&[Instant::now()]).is_none());
}

// ── ThinkSplitter state machine ─────────────────────────────────────

/// Run a slice of pieces through a splitter and collect the per-step
/// (visible_text, is_thinking) pairs. Convenience helper for the
/// table-driven cases below.
fn run_splitter(mut sm: ThinkSplitter, pieces: &[&str]) -> Vec<(String, bool)> {
    pieces.iter().map(|p| sm.step(p)).collect()
}

/// Non-reasoning models construct the splitter as `None`, so step_fn
/// short-circuits and is_thinking is always false. We verify the
/// default `new()` constructor behaves the same way: no tag in the
/// stream → every piece on the content channel.
#[test]
fn splitter_plain_text_stays_on_content_channel() {
    let out = run_splitter(ThinkSplitter::new(), &["Hello", " world"]);
    assert_eq!(out, vec![("Hello".into(), false), (" world".into(), false)]);
}

/// Qwen3 case (canonical): template prefills `<think>\n` into the
/// assistant turn, model emits reasoning text directly, then the
/// literal `</think>` token, then the final answer.
#[test]
fn splitter_qwen3_prefilled_routes_thinking_then_content() {
    let out = run_splitter(
        ThinkSplitter::new_qwen3_prefilled(),
        &["Let me think...", "</think>", "The answer", " is 221."],
    );
    assert_eq!(
        out,
        vec![
            ("Let me think...".to_owned(), true),
            (String::new(), false),
            ("The answer".to_owned(), false),
            (" is 221.".to_owned(), false),
        ]
    );
}

/// Some templates emit `<think>` as a regular token (no prefill).
/// Default-init splitter must flip to thinking on encounter, then
/// back on `</think>`.
#[test]
fn splitter_inline_open_close() {
    let out = run_splitter(
        ThinkSplitter::new(),
        &["<think>", "deliberate", "</think>", "answer"],
    );
    assert_eq!(
        out,
        vec![
            (String::new(), true),
            ("deliberate".to_owned(), true),
            (String::new(), false),
            ("answer".to_owned(), false),
        ]
    );
}

/// `</think>` literal embedded mid-piece (rare BPE case). The tag is
/// stripped; the visible text is concatenated and reported under the
/// channel after the transition (dominant-channel approximation,
/// documented in `ThinkSplitter::step`).
#[test]
fn splitter_close_tag_midpiece() {
    let mut sm = ThinkSplitter::new_qwen3_prefilled();
    let (text, is_thinking) = sm.step("thinking text</think>answer");
    assert_eq!(text, "thinking textanswer");
    assert!(!is_thinking);
    // Subsequent piece stays on content channel.
    let (text2, is_thinking2) = sm.step(" more");
    assert_eq!(text2, " more");
    assert!(!is_thinking2);
}

/// Empty visible piece after stripping the lone tag token → caller
/// should NOT emit an SSE delta event. The state machine returns
/// `(empty, new_state)`; the OpenAI / Anthropic SSE handlers check
/// `piece.is_empty()` and skip.
#[test]
fn splitter_lone_close_tag_emits_empty_visible() {
    let mut sm = ThinkSplitter::new_qwen3_prefilled();
    let (text, is_thinking) = sm.step("</think>");
    assert_eq!(text, "");
    assert!(!is_thinking);
}

// ── per-request thinking budget ───────────────────────────────────

/// PART 1: budget = 5 thinking pieces. Feed 6 thinking-channel pieces;
/// `force_close` (drained via `take_force_close`) must fire on the 6th —
/// the first piece that pushes the count past the budget — and exactly
/// once.
#[test]
fn splitter_budget_force_close_fires_on_overflow() {
    // enable_thinking=true → starts open (thinking channel); budget = 5.
    let mut sm = ThinkSplitter::new_for_request(true, Some(5), None, None);
    // Pieces 1..=5 are within budget → no force-close.
    for i in 1..=5 {
        let (_text, is_thinking) = sm.step("reason");
        assert!(is_thinking, "piece {i} should route to thinking channel");
        assert!(
            !sm.take_force_close(),
            "force_close must NOT fire within budget (piece {i})"
        );
    }
    // 6th thinking piece exceeds the budget of 5 → force_close fires.
    let (_text, is_thinking) = sm.step("reason");
    assert!(
        is_thinking,
        "6th piece is still emitted on thinking channel"
    );
    assert!(
        sm.take_force_close(),
        "force_close must fire on the 6th thinking piece (budget=5)"
    );
    // One-shot: a second take returns false even though budget stays exceeded.
    assert!(
        !sm.take_force_close(),
        "force_close is one-shot — must not re-fire after being taken"
    );
}

/// No budget set → `take_force_close` never fires regardless of how many
/// thinking pieces are emitted (zero-overhead default path).
#[test]
fn splitter_no_budget_never_force_closes() {
    let mut sm = ThinkSplitter::new_for_request(true, None, None, None);
    for _ in 0..50 {
        let _ = sm.step("reason");
        assert!(!sm.take_force_close());
    }
}

/// A prompt whose generation suffix CLOSED the think block must start the
/// splitter in answer-mode, so the first piece routes to the CONTENT channel
/// (`is_thinking=false`), not reasoning.
#[test]
fn splitter_closed_prompt_starts_on_content() {
    let mut sm = ThinkSplitter::new_for_request(false, None, None, None);
    let (text, is_thinking) = sm.step("Hello");
    assert_eq!(text, "Hello");
    assert!(
        !is_thinking,
        "a closed prompt must route the first piece to content"
    );
}

/// A prompt whose generation suffix LEFT the block open starts on the thinking
/// channel, exactly like the canonical `new_qwen3_prefilled`.
#[test]
fn splitter_open_prompt_starts_on_thinking() {
    let mut sm = ThinkSplitter::new_for_request(true, None, None, None);
    let (text, is_thinking) = sm.step("reason");
    assert_eq!(text, "reason");
    assert!(
        is_thinking,
        "an open prompt must start on the thinking channel"
    );
}

// ── per-request delimiter overrides ───────────────────────────────

/// Custom `thinking_end_token`: a splitter pre-opened with a custom
/// close delimiter routes pre-delimiter text to reasoning and
/// post-delimiter text to content, stripping the custom tag.
///
/// Verifies that `ThinkSplitter::step` uses the field-stored delimiter
/// strings rather than the hardcoded `"<think>"`/`"</think>"` literals.
#[test]
fn splitter_custom_end_token_routes_correctly() {
    // Simulate a caller that uses "</custom>" as the close delimiter.
    // The splitter starts open (prefilled start token already consumed).
    let mut sm = ThinkSplitter::new_for_request(
        true,
        None,
        Some("<custom_think>".to_owned()),
        Some("</custom>".to_owned()),
    );

    // Reasoning text before the custom end delimiter.
    let (text, is_thinking) = sm.step("deep thought");
    assert_eq!(text, "deep thought");
    assert!(is_thinking, "pre-delimiter text must route to reasoning");

    // Custom close delimiter is stripped; state flips to content.
    let (text, is_thinking) = sm.step("</custom>");
    assert!(text.is_empty(), "the delimiter itself must be stripped");
    assert!(!is_thinking, "post-delimiter state must be content");

    // Subsequent pieces go to content.
    let (text, is_thinking) = sm.step("answer text");
    assert_eq!(text, "answer text");
    assert!(
        !is_thinking,
        "subsequent pieces must stay on content channel"
    );

    // Default delimiters must NOT trigger transitions for this splitter.
    let (text, is_thinking) = sm.step("</think>");
    assert_eq!(
        text, "</think>",
        "default end delimiter must be treated as plain text"
    );
    assert!(
        !is_thinking,
        "default delimiter must not flip state on custom-delimiter splitter"
    );
}

/// Custom `thinking_start_token`: a splitter starting closed flips to
/// reasoning on the custom open delimiter (not on the default `"<think>"`).
#[test]
fn splitter_custom_start_token_routes_correctly() {
    let mut sm = ThinkSplitter::new_for_request(
        false,
        None,
        Some("<custom_think>".to_owned()),
        Some("</custom>".to_owned()),
    );

    // Default open delimiter must be treated as plain text.
    let (text, is_thinking) = sm.step("<think>");
    assert_eq!(text, "<think>", "default start delimiter is plain text");
    assert!(
        !is_thinking,
        "default start delimiter must not open thinking"
    );

    // Custom open delimiter flips to reasoning.
    let (text, is_thinking) = sm.step("<custom_think>");
    assert!(text.is_empty(), "custom start delimiter must be stripped");
    assert!(
        is_thinking,
        "custom start delimiter must open thinking channel"
    );

    // Reasoning text.
    let (text, is_thinking) = sm.step("thoughts");
    assert_eq!(text, "thoughts");
    assert!(is_thinking);

    // Custom close flips back to content.
    let (text, is_thinking) = sm.step("</custom>");
    assert_eq!(text, "");
    assert!(!is_thinking);
}

// ── Phase enum + per-phase metric emission ────────────────────────

#[test]
fn phase_enum_derives_equality_and_debug() {
    assert_eq!(Phase::Prefill, Phase::Prefill);
    assert_eq!(Phase::Decode, Phase::Decode);
    assert_ne!(Phase::Prefill, Phase::Decode);
    // Debug should produce a non-empty, recognisable string.
    let s = format!("{:?}", Phase::Prefill);
    assert!(s.contains("Prefill"), "Debug for Prefill: {s}");
    let s = format!("{:?}", Phase::Decode);
    assert!(s.contains("Decode"), "Debug for Decode: {s}");
}

#[test]
fn phase_is_copy_and_hash() {
    // Compile-time witness for Copy + Hash trait bounds via use sites.
    fn assert_copy<T: Copy>() {}
    fn assert_hash<T: std::hash::Hash>() {}
    assert_copy::<Phase>();
    assert_hash::<Phase>();
}

/// a single call to `record_itl_percentiles` writes exactly 6 rows
/// — 3 `itl_*_ms` (legacy) + 3 `tpot_*_ms` (new) — to the events table,
/// each with the correct op name, value, and unit. tpot rows mirror itl
/// rows numerically in v1.
#[test]
fn record_itl_percentiles_writes_six_rows_including_tpot() {
    use rusqlite::params;
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "tpot").expect("open");

    record_itl_percentiles(&rec, "test-model", "k8v8", 7.5, 12.0, 18.3);

    let conn = rmlx_metrics::schema::open(&db).expect("reopen");

    // Expect 6 rows total from this call.
    let total: i64 = conn
        .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))
        .expect("count all");
    // invariant: events table starts empty; only this call writes rows.
    assert_eq!(total, 6, "expected 6 rows (3 itl + 3 tpot), got {total}");

    for (op, expected) in [
        ("itl_p50_ms", 7.5),
        ("itl_p95_ms", 12.0),
        ("itl_p99_ms", 18.3),
        ("tpot_p50_ms", 7.5),
        ("tpot_p95_ms", 12.0),
        ("tpot_p99_ms", 18.3),
    ] {
        let (val, unit): (f64, String) = conn
            .query_row(
                "SELECT value, value_unit FROM events WHERE op = ?1 LIMIT 1",
                params![op],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap_or_else(|e| panic!("row for {op}: {e}"));
        assert!(
            (val - expected).abs() < 1e-6,
            "{op} value {val} != expected {expected}"
        );
        assert_eq!(unit, "ms", "{op} unit");
    }
}

/// registry exposes all four new ops with `ms` + lower_better.
#[test]
fn registry_exposes_new_ops() {
    for op in [
        "prefill_duration_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
    ] {
        let (unit, dir) =
            rmlx_metrics::registry::lookup(op).unwrap_or_else(|e| panic!("lookup {op}: {e}"));
        assert_eq!(unit, "ms", "{op} unit");
        assert_eq!(
            dir,
            rmlx_metrics::registry::Direction::LowerBetter,
            "{op} direction"
        );
    }
}

// ── Per-request KV-config override parsing ────────────────────────────────────

/// `parse_request_kv_quant` mirrors the `--kv-quant` CLI grammar: `"auto"` →
/// `None` (the request names no codec and runs the launch-resolved one), canonical
/// codec strings → the matching `KvQuant`, `"mixed"` → the canonical short
/// alias, and a malformed string → `Err` (the route maps this to HTTP 400).
#[test]
fn parse_request_kv_quant_maps_strings() {
    use rmlx_kv_quant::KvQuant;

    // "auto" → None (defer to launch/auto policy).
    assert_eq!(parse_request_kv_quant("auto"), Ok(None));
    // Case-insensitive + whitespace-tolerant.
    assert_eq!(parse_request_kv_quant("  AUTO "), Ok(None));

    // Canonical codec strings round-trip to the right variant.
    assert_eq!(parse_request_kv_quant("none"), Ok(Some(KvQuant::None)));
    assert_eq!(parse_request_kv_quant("bf16"), Ok(Some(KvQuant::None)));
    assert_eq!(parse_request_kv_quant("k8v4"), Ok(Some(KvQuant::K8V4)));
    assert_eq!(parse_request_kv_quant("k8v8"), Ok(Some(KvQuant::K8V8)));
    assert_eq!(parse_request_kv_quant("planar"), Ok(Some(KvQuant::Planar)));

    // "mixed" → the canonical Mixed{k8,v4,g64} short alias.
    assert_eq!(
        parse_request_kv_quant("mixed"),
        Ok(Some(KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }))
    );

    // Garbage → Err (route layer returns HTTP 400).
    assert!(parse_request_kv_quant("definitely-not-a-codec").is_err());
}

/// Default-when-absent: the override fields on a freshly built request are
/// `None`, so the generator falls through to its launch default (zero
/// regression for requests that omit `kv_quant` / `max_ctx`).
#[test]
fn generation_request_overrides_default_none() {
    let req = sample_req();
    assert_eq!(req.kv_quant_override, None);
    assert_eq!(req.max_ctx_override, None);
}

/// A per-request override is carried verbatim on the `GenerationRequest` so the
/// generator's `generate()` can prefer it over the launch default.
#[test]
fn generation_request_carries_overrides() {
    let mut req = sample_req();
    req.kv_quant_override = Some(rmlx_kv_quant::KvQuant::K8V4);
    req.max_ctx_override = Some(8192);
    assert_eq!(req.kv_quant_override, Some(rmlx_kv_quant::KvQuant::K8V4));
    assert_eq!(req.max_ctx_override, Some(8192));
}

// ── initial think-channel is read off the rendered prompt ───────────────────

/// Assistant-turn tails the three test-target chat templates actually render.
/// The point of the table is that they disagree *within* the set of
/// thinking-capable architectures, which is why the channel cannot be derived
/// from the architecture or from `enable_thinking`.
const TAIL_CLOSED_PREFILL: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
const TAIL_OPEN_PREFILL: &str = "<|im_start|>assistant\n<think>\n";
const TAIL_NO_PREFILL: &str = "<|turn>model\n";

#[test]
fn prompt_think_open_reads_the_rendered_prefill() {
    let cases: &[(&str, bool, &str)] = &[
        (
            TAIL_CLOSED_PREFILL,
            false,
            "template prefills a CLOSED think block — the model answers directly",
        ),
        (
            TAIL_OPEN_PREFILL,
            true,
            "template prefills an OPEN think block — the model resumes reasoning",
        ),
        (
            TAIL_NO_PREFILL,
            false,
            "template prefills nothing — the model opens its own block if it wants one",
        ),
        (
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nwhy\n</think>\n\nbecause<|im_end|>\n<|im_start|>assistant\n<think>\n",
            true,
            "closed history blocks must not mask the final open one",
        ),
        (
            "<|im_start|>assistant\n<think>\na\n</think>\n\nb<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            false,
            "last delimiter wins across a multi-turn transcript",
        ),
    ];
    for (rendered, want, why) in cases {
        assert_eq!(
            prompt_leaves_think_open(rendered, "<think>", "</think>"),
            *want,
            "{why}"
        );
    }
}

#[test]
fn prompt_think_open_honours_delimiter_overrides() {
    assert!(prompt_leaves_think_open(
        "…<|reason|>",
        "<|reason|>",
        "<|/reason|>"
    ));
    assert!(!prompt_leaves_think_open(
        "…<|reason|><|/reason|>",
        "<|reason|>",
        "<|/reason|>"
    ));
    // Default delimiters must not fire on a custom-delimiter transcript.
    assert!(!prompt_leaves_think_open(
        "…<|reason|>",
        "<think>",
        "</think>"
    ));
}

/// The defect this replaces: a splitter that starts open on a prompt whose
/// template already CLOSED the think block never sees a `</think>`, so every
/// piece is reported as reasoning for the whole request. Downstream that
/// latches the constraint engine's `is_thinking` gate and the schema is never
/// enforced. Driving the initial channel from the prompt closes it.
#[test]
fn closed_prefill_starts_the_splitter_in_answer_mode() {
    let pieces = ["{", "\"v\"", ":", "\"x\"", "}"];

    let open = prompt_leaves_think_open(TAIL_CLOSED_PREFILL, "<think>", "</think>");
    assert!(!open, "a closed prefill leaves no open think block");
    let out = run_splitter(
        ThinkSplitter::new_for_request(open, None, None, None),
        &pieces,
    );
    assert!(
        out.iter().all(|(_, thinking)| !*thinking),
        "every piece must be reported on the content channel: {out:?}"
    );

    // Contrast: assuming "open" — the pre-fix derivation for any
    // thinking-capable arch — latches the reasoning channel forever.
    let latched = run_splitter(
        ThinkSplitter::new_for_request(true, None, None, None),
        &pieces,
    );
    assert!(
        latched.iter().all(|(_, thinking)| *thinking),
        "assumed-open splitter reports the whole answer as reasoning: {latched:?}"
    );
}

/// An open prefill must still start in the reasoning channel and flip on the
/// model's own `</think>` — the prompt-derived channel must not regress the
/// architecture it already worked for.
#[test]
fn open_prefill_still_starts_the_splitter_in_reasoning_mode() {
    let open = prompt_leaves_think_open(TAIL_OPEN_PREFILL, "<think>", "</think>");
    assert!(open, "an open prefill leaves the think block open");
    let out = run_splitter(
        ThinkSplitter::new_for_request(open, None, None, None),
        &["reasoning", "</think>", "answer"],
    );
    assert_eq!(
        out,
        vec![
            ("reasoning".to_owned(), true),
            (String::new(), false),
            ("answer".to_owned(), false),
        ]
    );
}

// ── Usage accounting ─────────────────────────────────────────

fn token_item(piece: &str, done: bool) -> GenerationToken {
    GenerationToken {
        token_id: 7,
        piece: piece.to_owned(),
        done,
        finish_reason: done.then(|| "stop".to_owned()),
        is_thinking: false,
        logprobs: None,
    }
}

/// A decode step counts, also when it adds no text (the end-of-turn token);
/// the terminal item does not, also when it carries text.
#[test]
fn the_terminal_item_alone_is_not_counted_toward_usage() {
    assert!(token_item("hi", false).counts_toward_usage());
    assert!(token_item("", false).counts_toward_usage());
    assert!(!token_item("", true).counts_toward_usage());
    assert!(!token_item("\u{FFFD}", true).counts_toward_usage());
}

/// The terminal item carries the finish reason and the withheld tail, on the
/// content channel, and is not counted.
#[test]
fn the_terminal_item_carries_the_tail_and_the_finish_reason() {
    let item = GenerationToken::terminal(7, "length".to_owned(), "\u{FFFD}".to_owned());
    assert!(item.done);
    assert_eq!(item.piece, "\u{FFFD}");
    assert_eq!(item.finish_reason.as_deref(), Some("length"));
    assert!(!item.is_thinking);
    assert!(!item.counts_toward_usage());
}

/// The reply stream of a job owns the job: the stop flag of the job is clear
/// while the reply is read, and a drop of the stream sets it. A stream that
/// lets go of the job at once stops every request before its first token.
#[tokio::test(flavor = "multi_thread")]
async fn the_reply_stream_of_a_job_sets_the_stop_flag_only_when_it_is_dropped() {
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Duration;

    let (tx, rx) = tokio::sync::mpsc::channel::<bool>(4);
    let (saw_tx, saw_rx) = mpsc::channel();
    let job = run_admitted(None, move |cancel| {
        for _ in 0..3 {
            tx.blocking_send(cancel.load(Relaxed)).ok();
        }
        let waited = Instant::now();
        while !cancel.load(Relaxed) && waited.elapsed() < Duration::from_secs(2) {
            std::thread::sleep(Duration::from_millis(1));
        }
        saw_tx.send(cancel.load(Relaxed)).ok();
    });
    let mut reply = Box::pin(job_stream(rx, job));
    for item in 0..3 {
        assert_eq!(
            reply.next().await,
            Some(false),
            "item {item}: the stop flag of a job whose reply is read"
        );
    }
    drop(reply);
    let saw = tokio::task::spawn_blocking(move || saw_rx.recv_timeout(Duration::from_secs(5)))
        .await
        .expect("the report task does not panic");
    assert_eq!(saw, Ok(true), "the job did not see the stop of its reply");
}

/// The reply stream ends when its job ends: it gives each item the job sent,
/// then `None`.
#[tokio::test(flavor = "multi_thread")]
async fn the_reply_stream_of_a_job_ends_when_the_job_ends() {
    let (tx, rx) = tokio::sync::mpsc::channel::<u32>(4);
    let job = run_admitted(None, move |_cancel| {
        for item in 0..2 {
            tx.blocking_send(item).ok();
        }
    });
    let reply: Vec<u32> = job_stream(rx, job).collect().await;
    assert_eq!(reply, [0, 1]);
}

// ── run_admitted: the stop flag and the GPU permit of a blocking job ─────────

use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::mpsc;
use std::time::Duration;

use tokio::sync::Semaphore;

/// How long a scripted job waits for the flag before it gives up and reports
/// that it never saw it. A job that never stops would hang the test binary.
pub(crate) const FLAG_WAIT: Duration = Duration::from_secs(2);

/// How long the test waits for a message from a scripted job.
pub(crate) const REPORT_WAIT: Duration = Duration::from_secs(5);

/// The one-permit GPU queue and its in-flight gauge.
pub(crate) fn gpu_queue() -> (Arc<Semaphore>, Arc<AtomicUsize>) {
    (Arc::new(Semaphore::new(1)), Arc::new(AtomicUsize::new(0)))
}

pub(crate) async fn admitted(queue: &Arc<Semaphore>, pending: &Arc<AtomicUsize>) -> GpuAdmission {
    match admit_request(queue, pending, 64).await {
        Admission::Admitted { guard, .. } => guard,
        Admission::QueueFull => panic!("an empty queue admits"),
    }
}

/// A scripted walk: it decodes "windows" of 1 ms until the flag is set or
/// `FLAG_WAIT` passes, and returns whether it saw the flag.
pub(crate) fn wait_for_flag(cancel: &AtomicBool) -> bool {
    let start = Instant::now();
    while !cancel.load(Ordering::Relaxed) && start.elapsed() < FLAG_WAIT {
        std::thread::sleep(Duration::from_millis(1));
    }
    cancel.load(Ordering::Relaxed)
}

/// Wait on a std channel from async code without blocking the runtime.
pub(crate) async fn report<T: Send + 'static>(
    rx: mpsc::Receiver<T>,
) -> Result<T, mpsc::RecvTimeoutError> {
    tokio::task::spawn_blocking(move || rx.recv_timeout(REPORT_WAIT))
        .await
        .expect("the report task does not panic")
}

/// The request timeout drops the request future, as `timeout_mw` does: the
/// job sees the flag and stops.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_timeout_sets_the_flag_and_the_job_stops() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let (tx, rx) = mpsc::channel();
    let request = run_admitted(Some(guard), move |cancel| {
        tx.send(wait_for_flag(cancel)).ok();
    });
    let timed_out = tokio::time::timeout(Duration::from_millis(50), request).await;
    assert!(timed_out.is_err(), "the scripted job outlives the timeout");
    assert_eq!(report(rx).await, Ok(true), "the job did not see the flag");
}

/// After the timeout the permit stays with the job until the job returns;
/// then it is free and the gauge is back to 0.
#[tokio::test(flavor = "multi_thread")]
async fn the_permit_stays_held_until_the_stopped_job_returns() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let (saw_tx, saw_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let request = run_admitted(Some(guard), move |cancel| {
        saw_tx.send(wait_for_flag(cancel)).ok();
        release_rx.recv_timeout(REPORT_WAIT).ok();
    });
    let timed_out = tokio::time::timeout(Duration::from_millis(50), request).await;
    assert!(timed_out.is_err());
    assert_eq!(
        report(saw_rx).await,
        Ok(true),
        "the job did not see the flag"
    );

    assert_eq!(
        queue.available_permits(),
        0,
        "the permit was freed while the job still runs"
    );
    assert_eq!(pending.load(Ordering::Acquire), 1);

    release_tx.send(()).expect("the job waits for the release");
    let permit = tokio::time::timeout(REPORT_WAIT, queue.acquire()).await;
    assert!(
        matches!(permit, Ok(Ok(_))),
        "the permit is not free after the job returned"
    );
    assert_eq!(pending.load(Ordering::Acquire), 0);
}

/// A job that runs to its end never sees the flag, gives its result, and
/// frees the permit.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_ends_sees_no_flag_and_frees_the_permit() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let saw_flag = run_admitted(Some(guard), |cancel| cancel.load(Ordering::Relaxed)).await;
    assert!(matches!(saw_flag, Ok(false)), "{saw_flag:?}");
    assert_eq!(queue.available_permits(), 1);
    assert_eq!(pending.load(Ordering::Acquire), 0);
}

/// The stop of a request that is gone is not the stop of the next request: the
/// next job starts with its flag clear and runs to its end.
#[tokio::test(flavor = "multi_thread")]
async fn the_stop_of_one_request_is_not_the_stop_of_the_next() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let (saw_tx, saw_rx) = mpsc::channel();
    let gone = run_admitted(Some(guard), move |cancel| {
        saw_tx.send(wait_for_flag(cancel)).ok();
    });
    let timed_out = tokio::time::timeout(Duration::from_millis(50), gone).await;
    assert!(timed_out.is_err());
    assert_eq!(
        report(saw_rx).await,
        Ok(true),
        "the first job did not see its flag"
    );

    let guard = admitted(&queue, &pending).await;
    let next = run_admitted(Some(guard), |cancel| cancel.load(Ordering::Relaxed)).await;
    assert!(matches!(next, Ok(false)), "{next:?}");
}

/// A job that panics frees the permit, and the caller gets the panic as an
/// error.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_panics_frees_the_permit() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let result = run_admitted(Some(guard), |_| -> u8 { panic!("scripted job panic") }).await;
    assert!(result.as_ref().is_err_and(tokio::task::JoinError::is_panic));
    assert_eq!(queue.available_permits(), 1);
    assert_eq!(pending.load(Ordering::Acquire), 0);
}
