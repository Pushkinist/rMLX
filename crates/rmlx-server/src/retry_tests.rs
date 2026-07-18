use super::*;
use rmlx_core::{Error as E, OomPhase};
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::engine::{GenerationToken, SamplingParams};

// ── classify() ───────────────────────────────────────────────────────────

#[test]
fn classify_mlx_is_migratable() {
    let err = E::Mlx("Metal watchdog killed process".to_owned());
    assert_eq!(classify(&err), RetryClass::Migratable);
}

#[test]
fn classify_other_is_migratable() {
    let err = E::Other("task panicked during decode".to_owned());
    assert_eq!(classify(&err), RetryClass::Migratable);
}

#[test]
fn classify_smoke_probe_is_fatal() {
    let err = E::SmokeProbe("NaN logits".to_owned());
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_oom_generation_is_fatal() {
    let err = E::Oom {
        phase: OomPhase::Generation,
        requested_bytes: None,
        peak_alloc_mb: None,
        msg: "out of memory".to_owned(),
    };
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_oom_load_weights_is_fatal() {
    let err = E::Oom {
        phase: OomPhase::LoadWeights,
        requested_bytes: Some(1_000_000),
        peak_alloc_mb: Some(16),
        msg: "weight OOM".to_owned(),
    };
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_oom_kv_cache_is_fatal() {
    let err = E::Oom {
        phase: OomPhase::LoadKvCache,
        requested_bytes: None,
        peak_alloc_mb: None,
        msg: "kv OOM".to_owned(),
    };
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_config_is_fatal() {
    let err = E::Config("missing field".to_owned());
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_loader_is_fatal() {
    let err = E::Loader("safetensors mismatch".to_owned());
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_quant_is_fatal() {
    let err = E::Quant("unsupported quant".to_owned());
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_model_is_fatal() {
    let err = E::Model("arch unknown".to_owned());
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_io_is_fatal() {
    let err = E::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "file not found",
    ));
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only string construction; no fallible ops"
)]
fn classify_speculative_pairing_is_fatal() {
    // SpeculativePairing is emitted when --draft-kind mtp is given an
    // incompatible draft snapshot (e.g. plain Gemma4 fed to the Qwen MTP
    // loader). Retrying would hit the same rejection — must be Fatal, not
    // Migratable.
    let err = E::SpeculativePairing {
        reason: "draft model architecture 'Gemma4ForConditionalGeneration' is not a supported \
                 --draft-kind mtp family"
            .to_owned(),
    };
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_kv_ceiling_exceeded_is_fatal() {
    // A request that overruns the `--max-ctx` ceiling hits the same ceiling on
    // every attempt — whichever phase (prefill or decode) crossed it — so a
    // replay is futile. Must be Fatal, never Migratable.
    let err = E::KvCeilingExceeded {
        requested: 641,
        ceiling: 640,
    };
    assert_eq!(classify(&err), RetryClass::Fatal);
}

#[test]
fn classify_kv_hard_cap_exceeded_is_fatal() {
    let err = E::KvHardCapExceeded {
        requested: 4097,
        cap: 4096,
    };
    assert_eq!(classify(&err), RetryClass::Fatal);
}

// ── is_replayable() ──────────────────────────────────────────────────────

fn req_with_temp(temperature: f32) -> GenerationRequest {
    GenerationRequest {
        model_id: "mock".to_owned(),
        prompt_tokens: vec![],
        max_tokens: 100,
        sampling: SamplingParams {
            temperature,
            ..SamplingParams::default()
        },
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
        enable_thinking: None,
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

#[test]
fn is_replayable_all_clear_returns_true() {
    assert!(is_replayable(&req_with_temp(0.0), true));
}

#[test]
fn is_replayable_temperature_positive_returns_false() {
    assert!(!is_replayable(&req_with_temp(0.5), true));
}

#[test]
fn is_replayable_temperature_one_returns_false() {
    assert!(!is_replayable(&req_with_temp(1.0), true));
}

#[test]
fn is_replayable_n_greater_than_one_returns_false() {
    assert!(!is_replayable(&req_with_temp(0.0), false));
}

#[test]
fn is_replayable_guided_decoding_present_returns_false() {
    let mut req = req_with_temp(0.0);
    req.constraint = Some(Box::new(rmlx_models::NoOpConstraint::new()));
    assert!(!is_replayable(&req, true));
}

// ── MockGenerator ────────────────────────────────────────────────────────

/// Emits a deterministic token sequence, optionally injecting a Migratable
/// error after `fail_after` tokens on the specified attempt (0-indexed).
struct MockGenerator {
    tokens: Vec<u32>,
    fail_on_attempt: Option<(usize, usize)>,
    call_count: Arc<AtomicUsize>,
}

impl MockGenerator {
    fn always_ok(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            fail_on_attempt: None,
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn fail_attempt(tokens: Vec<u32>, attempt: usize, after: usize) -> Self {
        Self {
            tokens,
            fail_on_attempt: Some((attempt, after)),
            call_count: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Generator for MockGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
    {
        let attempt = self.call_count.fetch_add(1, Ordering::SeqCst);
        let tokens = self.tokens.clone();
        let fail_on_attempt = self.fail_on_attempt;

        let items: Vec<rmlx_core::Result<GenerationToken>> = tokens
            .iter()
            .enumerate()
            .flat_map(|(i, &token_id)| {
                if let Some((fail_attempt, fail_after)) = fail_on_attempt {
                    if attempt == fail_attempt && i == fail_after {
                        return vec![Err(rmlx_core::Error::Mlx(
                            "Metal watchdog (mock)".to_owned(),
                        ))];
                    }
                }
                vec![Ok(GenerationToken {
                    token_id,
                    piece: format!("t{token_id}"),
                    done: false,
                    finish_reason: None,
                    is_thinking: false,
                    logprobs: None,
                })]
            })
            .collect();
        Box::pin(futures::stream::iter(items))
    }
}

fn make_initial_req(prompt: Vec<u32>, max_tokens: u32) -> GenerationRequest {
    GenerationRequest {
        model_id: "mock".to_owned(),
        prompt_tokens: prompt,
        max_tokens,
        sampling: SamplingParams::default(),
        stop: vec![],
        stream: true,
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
        enable_thinking: None,
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

fn drive(
    gen: Arc<dyn Generator>,
    prompt: Vec<u32>,
    max_tokens: u32,
    max_retries: u32,
) -> ReplayStream {
    let initial = make_initial_req(prompt, max_tokens);
    let plan = RequestPlan::from_gen_req(&initial);
    replay_stream(gen, initial, plan, max_retries)
}

// ── replay_stream core scenarios ─────────────────────────────────────────

/// Happy path: seamless no-error stream emits all N tokens.
#[tokio::test]
async fn replay_stream_no_error_emits_all_tokens() {
    let tokens: Vec<u32> = (1..=5).collect();
    let gen = Arc::new(MockGenerator::always_ok(tokens.clone()));
    let mut s = drive(gen, vec![0u32], tokens.len() as u32, DEFAULT_MAX_RETRIES);
    let mut received = vec![];
    while let Some(item) = s.next().await {
        received.push(item.unwrap().token_id);
    }
    assert_eq!(received, tokens);
}

/// Migratable error after K tokens → client sees all N tokens, exactly
/// once, in order (seamless replay).
#[tokio::test]
async fn replay_stream_migratable_mid_stream_seamless() {
    let tokens: Vec<u32> = (10..20).collect(); // 10 tokens
    let fail_after = 4;
    let gen = Arc::new(MockGenerator::fail_attempt(tokens.clone(), 0, fail_after));
    let mut s = drive(
        gen,
        vec![0u32, 1u32, 2u32],
        tokens.len() as u32,
        DEFAULT_MAX_RETRIES,
    );
    let mut received = vec![];
    while let Some(item) = s.next().await {
        match item {
            Ok(tok) => received.push(tok.token_id),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
    assert_eq!(received, tokens, "client must see all tokens exactly once");
}

/// Fatal error: no retry — error forwarded immediately.
#[tokio::test]
async fn replay_stream_fatal_error_not_retried() {
    struct FatalGen;
    impl Generator for FatalGen {
        fn generate(
            &self,
            _req: GenerationRequest,
        ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
        {
            Box::pin(futures::stream::once(async {
                Err(rmlx_core::Error::SmokeProbe("NaN logits".to_owned()))
            }))
        }
    }
    let gen = Arc::new(FatalGen);
    let mut s = drive(gen, vec![1u32], 10, 2);
    let item = s.next().await.expect("stream must yield at least one item");
    assert!(item.is_err(), "fatal error must be forwarded as-is");
    assert!(s.next().await.is_none());
}

/// Client cancel: drop the stream before completion — no panic, no hang.
#[tokio::test]
async fn replay_stream_client_cancel_silent_exit() {
    let tokens: Vec<u32> = (1..=20).collect();
    let gen = Arc::new(MockGenerator::always_ok(tokens));
    let mut s = drive(gen, vec![0u32], 20, 2);
    let _ = s.next().await;
    drop(s);
    // Passes if it does not panic or hang.
}

/// Prefix divergence: retry returns a different token at the prefix position
/// — stream ends with a fatal error, the bad token is NOT delivered.
#[tokio::test]
async fn replay_stream_prefix_divergence_aborts() {
    // Diverge at position 2 (within the prefix that was already delivered).
    // Attempt 0 errors after token 3 (fail_after=3), so 3 tokens delivered.
    // Attempt 1 diverges at position 2 (inside prefix of length 3).
    struct DivergingGen {
        // attempt 0: emit tokens 0..N, error at index 3
        // attempt 1: emit tokens 0..2 correctly, then DIVERGE at 2
        call_count: Arc<AtomicUsize>,
        tokens: Vec<u32>,
    }
    impl Generator for DivergingGen {
        fn generate(
            &self,
            _req: GenerationRequest,
        ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
        {
            let attempt = self.call_count.fetch_add(1, Ordering::SeqCst);
            let tokens = self.tokens.clone();
            let items: Vec<rmlx_core::Result<GenerationToken>> = tokens
                .iter()
                .enumerate()
                .flat_map(|(i, &token_id)| {
                    if attempt == 0 && i == 3 {
                        return vec![Err(E::Mlx("watchdog".to_owned()))];
                    }
                    if attempt == 1 && i == 2 {
                        // Return wrong token at prefix position 2.
                        return vec![Ok(GenerationToken {
                            token_id: 0xDEAD,
                            piece: "DIVERGED".to_owned(),
                            done: false,
                            finish_reason: None,
                            is_thinking: false,
                            logprobs: None,
                        })];
                    }
                    vec![Ok(GenerationToken {
                        token_id,
                        piece: format!("t{token_id}"),
                        done: false,
                        finish_reason: None,
                        is_thinking: false,
                        logprobs: None,
                    })]
                })
                .collect();
            Box::pin(futures::stream::iter(items))
        }
    }

    let gen = Arc::new(DivergingGen {
        call_count: Arc::new(AtomicUsize::new(0)),
        tokens: (0u32..10).collect(),
    });
    let mut s = drive(gen, vec![99u32], 10, 2);
    // Collect results; we expect 3 good tokens then an error.
    let mut good = vec![];
    let mut got_error = false;
    while let Some(item) = s.next().await {
        if let Ok(tok) = item {
            good.push(tok.token_id)
        } else {
            got_error = true;
            break;
        }
    }
    assert!(got_error, "divergence must produce a fatal error");
    // The diverged token (0xDEAD) must NOT appear in delivered tokens.
    assert!(
        !good.contains(&0xDEAD),
        "diverged token must not be delivered"
    );
}

/// A real decode-step crash must NOT be laundered into a synthetic "prefix
/// divergence" message on replay. A deterministic decode crash (e.g. a KV
/// boundary reshape) reproduces at the same point on retry, so the replay's
/// first regenerated token differs from the delivered prefix and the divergence
/// guard fires. The error the client receives must still carry the REAL cause —
/// otherwise a finish_reason-keying caller cannot tell a crashed stream from a
/// clean short one.
#[tokio::test]
async fn replay_divergence_surfaces_real_crash_not_laundered_message() {
    // Attempt 0: emit 0,1,2 then a distinctive decode crash (delivers 3).
    // Attempt 1: diverge at prefix position 2 — the divergence guard fires.
    const CRASH_MSG: &str = "reshape: Cannot reshape array of size 0 into shape (1,8,1,32)";
    struct CrashThenDivergeGen {
        call_count: Arc<AtomicUsize>,
        tokens: Vec<u32>,
    }
    impl Generator for CrashThenDivergeGen {
        fn generate(
            &self,
            _req: GenerationRequest,
        ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
        {
            let attempt = self.call_count.fetch_add(1, Ordering::SeqCst);
            let tokens = self.tokens.clone();
            let items: Vec<rmlx_core::Result<GenerationToken>> = tokens
                .iter()
                .enumerate()
                .flat_map(|(i, &token_id)| {
                    if attempt == 0 && i == 3 {
                        return vec![Err(E::Mlx(CRASH_MSG.to_owned()))];
                    }
                    if attempt >= 1 && i == 2 {
                        return vec![Ok(GenerationToken {
                            token_id: 0xDEAD,
                            piece: "DIVERGED".to_owned(),
                            done: false,
                            finish_reason: None,
                            is_thinking: false,
                            logprobs: None,
                        })];
                    }
                    vec![Ok(GenerationToken {
                        token_id,
                        piece: format!("t{token_id}"),
                        done: false,
                        finish_reason: None,
                        is_thinking: false,
                        logprobs: None,
                    })]
                })
                .collect();
            Box::pin(futures::stream::iter(items))
        }
    }

    let gen = Arc::new(CrashThenDivergeGen {
        call_count: Arc::new(AtomicUsize::new(0)),
        tokens: (0u32..10).collect(),
    });
    let mut s = drive(gen, vec![99u32], 10, 2);
    let mut last_error: Option<String> = None;
    while let Some(item) = s.next().await {
        if let Err(e) = item {
            last_error = Some(e.to_string());
            break;
        }
    }
    let err = last_error.expect("a decode crash must reach the client as an error");
    assert!(
        err.contains("Cannot reshape array of size 0"),
        "the real decode crash must reach the client, not a laundered message; got: {err}"
    );
    assert!(
        !err.contains("prefix divergence"),
        "the retry envelope must not launder the crash into a synthetic \
         'prefix divergence' message; got: {err}"
    );
}

/// Underrun: retry terminates before reproducing the delivered prefix —
/// stream ends with an error.
#[tokio::test]
async fn replay_stream_underrun_aborts() {
    // Attempt 0: emit 6 tokens then error → delivers 5 (error at index 5).
    // Attempt 1 (retry): terminates after 3 tokens (underrun, skip_count=5).
    struct UnderrunGen {
        call_count: Arc<AtomicUsize>,
        tokens: Vec<u32>,
    }
    impl Generator for UnderrunGen {
        fn generate(
            &self,
            _req: GenerationRequest,
        ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
        {
            let attempt = self.call_count.fetch_add(1, Ordering::SeqCst);
            let tokens = self.tokens.clone();
            let items: Vec<rmlx_core::Result<GenerationToken>> = tokens
                .iter()
                .enumerate()
                .flat_map(|(i, &token_id)| {
                    if attempt == 0 && i == 5 {
                        return vec![Err(E::Mlx("watchdog".to_owned()))];
                    }
                    // On retry (attempt 1), stop at token 3 (truncated EOS).
                    if attempt == 1 && i == 3 {
                        return vec![];
                    }
                    vec![Ok(GenerationToken {
                        token_id,
                        piece: format!("t{token_id}"),
                        done: false,
                        finish_reason: None,
                        is_thinking: false,
                        logprobs: None,
                    })]
                })
                .collect();
            Box::pin(futures::stream::iter(items))
        }
    }

    let gen = Arc::new(UnderrunGen {
        call_count: Arc::new(AtomicUsize::new(0)),
        tokens: (0u32..10).collect(),
    });
    let mut s = drive(gen, vec![99u32], 10, 2);
    let mut got_error = false;
    while let Some(item) = s.next().await {
        if item.is_err() {
            got_error = true;
            break;
        }
    }
    assert!(got_error, "underrun must produce a fatal error");
}

/// Client-drop aborts the engine task — the generator stops producing
/// after stream drop.
#[tokio::test]
async fn replay_stream_client_drop_aborts_engine_task() {
    let counter = Arc::new(AtomicUsize::new(0));
    struct CountingGen {
        counter: Arc<AtomicUsize>,
    }
    impl Generator for CountingGen {
        fn generate(
            &self,
            _req: GenerationRequest,
        ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
        {
            let counter = Arc::clone(&self.counter);
            // Produce 100 tokens lazily via unfold, incrementing the counter
            // on each poll so the count reflects actual items yielded by the
            // stream, not pre-built Vec items.
            Box::pin(futures::stream::unfold(
                (0u32, counter),
                |(i, counter)| async move {
                    if i >= 100 {
                        return None;
                    }
                    counter.fetch_add(1, Ordering::SeqCst);
                    let tok = Ok(GenerationToken {
                        token_id: i,
                        piece: format!("t{i}"),
                        done: false,
                        finish_reason: None,
                        is_thinking: false,
                        logprobs: None,
                    });
                    Some((tok, (i + 1, counter)))
                },
            ))
        }
    }

    let gen = Arc::new(CountingGen {
        counter: Arc::clone(&counter),
    });
    let mut s = drive(gen, vec![0u32], 100, 2);
    // Consume just 3 tokens, then drop.
    for _ in 0..3 {
        let _ = s.next().await;
    }
    drop(s);
    // Give the task a moment to process the abort.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let after_drop = counter.load(Ordering::SeqCst);
    // The stream sends into a channel of cap 4; the task may have
    // buffered at most 3 consumed + 4 channel cap = 7 before blocking
    // on the next send (which fails because rx is dropped). Must be
    // well under 100.
    assert!(
        after_drop <= 10,
        "engine task must stop after stream drop (produced {after_drop})"
    );
}

// ── generate-call count tests (proxy for single-emit invariant) ───
//
// The real kv_cache_bytes / itl_p*_ms writes happen inside the blocking
// engine thread, gated on `steps_result.is_ok()`. MockGenerator never
// touches EventRecorder. These tests verify the attempt count as the proxy:
// correct count → only the successful attempt could have emitted.

/// Happy path (0 retries): generate called exactly once.
#[tokio::test]
async fn row_count_happy_path_one_attempt() {
    let tokens: Vec<u32> = (1..=5).collect();
    let gen = Arc::new(MockGenerator::always_ok(tokens.clone()));
    let call_count = Arc::clone(&gen.call_count);

    let mut s = drive(gen, vec![0u32], tokens.len() as u32, DEFAULT_MAX_RETRIES);
    while let Some(item) = s.next().await {
        item.unwrap();
    }
    // Happy path: exactly 1 call to generate.
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        1,
        "happy path must call generate exactly once"
    );
}

#[tokio::test]
async fn row_count_one_retry_two_attempts() {
    let tokens: Vec<u32> = (1..=8).collect();
    let fail_after = 3;
    let gen = Arc::new(MockGenerator::fail_attempt(tokens.clone(), 0, fail_after));
    let call_count = Arc::clone(&gen.call_count);

    let mut s = drive(gen, vec![0u32], tokens.len() as u32, DEFAULT_MAX_RETRIES);
    while let Some(item) = s.next().await {
        let _ = item;
    }
    // 1 retry: 2 calls total.
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        2,
        "1-retry path must call generate exactly twice"
    );
}

#[tokio::test]
async fn row_count_all_retries_exhausted_three_attempts() {
    // Fail on every attempt: attempt 0 errors at 3, attempt 1 errors at 3,
    // attempt 2 also errors at 3 → retries exhausted, stream ends in error.
    struct AlwaysFailGen {
        call_count: Arc<AtomicUsize>,
        tokens: Vec<u32>,
    }
    impl Generator for AlwaysFailGen {
        fn generate(
            &self,
            _req: GenerationRequest,
        ) -> Pin<Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>>
        {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let tokens = self.tokens.clone();
            let items: Vec<rmlx_core::Result<GenerationToken>> = tokens
                .iter()
                .enumerate()
                .flat_map(|(i, &token_id)| {
                    if i == 3 {
                        return vec![Err(E::Mlx("watchdog".to_owned()))];
                    }
                    vec![Ok(GenerationToken {
                        token_id,
                        piece: format!("t{token_id}"),
                        done: false,
                        finish_reason: None,
                        is_thinking: false,
                        logprobs: None,
                    })]
                })
                .collect();
            Box::pin(futures::stream::iter(items))
        }
    }

    let call_count = Arc::new(AtomicUsize::new(0));
    let gen = Arc::new(AlwaysFailGen {
        call_count: Arc::clone(&call_count),
        tokens: (0u32..10).collect(),
    });

    let mut s = drive(gen, vec![0u32], 10, DEFAULT_MAX_RETRIES);
    let mut final_item = None;
    while let Some(item) = s.next().await {
        final_item = Some(item);
    }
    // Stream must end in error.
    assert!(
        final_item.is_some_and(|i| i.is_err()),
        "exhausted retries must produce an error"
    );
    // max_retries=2 → 3 total attempts.
    assert_eq!(
        call_count.load(Ordering::SeqCst),
        3,
        "exhausted-retry path must call generate 3 times"
    );
}
