//! Speculative dispatcher unit tests.

use super::*;

use std::time::{Duration, Instant};

/// Compile-check: ensure the public type and methods exist with
/// the expected signatures. No runtime work.
#[test]
fn dispatcher_module_compiles() {
    fn _assert_signatures() {
        let _: fn(&Path, &Path, Device) -> Result<SpeculativeDispatcher> =
            SpeculativeDispatcher::load_speculative;
        let _: fn(&Path, Device) -> Result<SpeculativeDispatcher> =
            SpeculativeDispatcher::load_verifier_only;
        let _: fn(&SpeculativeDispatcher, &[u32], usize) -> Result<Array> =
            SpeculativeDispatcher::spec_forward;
    }
    _assert_signatures();
}

/// A sidecar drafter costs one resident copy of the verifier, not two.
///
/// `load_verifier_only` is the constructor the MTP / EAGLE-3 / DFlash serve
/// branches take; the empty draft slot is what makes the second `load_model`
/// impossible rather than merely unused. The two-model generators must then
/// say so rather than silently drafting with the verifier.
///
/// Loads on `Device::Cpu` and dispatches no Metal, so it is not `#[ignore]`d —
/// the snapshot guards below skip it cleanly when Open Models is absent.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: a snapshot this process has already checked for existence but cannot load is a broken checkout, and the panic names it"
)]
fn load_verifier_only_holds_no_draft_model() {
    let Some(path_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E2B").map(std::path::PathBuf::from)
    else {
        eprintln!("[spec_test] skipping: RMLX_TEST_MODEL_GEMMA4_E2B not set");
        return;
    };
    let path = path_buf.as_path();
    if !path.exists() {
        eprintln!("[spec_test] snapshot absent — skipping");
        return;
    }

    let disp =
        SpeculativeDispatcher::load_verifier_only(path, Device::Cpu).expect("load_verifier_only");
    assert_eq!(disp.vocab_size(), disp.verifier.vocab_size());

    let msg = disp
        .draft_model()
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        msg.contains("needs a draft model"),
        "sidecar dispatcher must hold no draft model; got: {msg:?}"
    );
}

/// `load_speculative` refuses a verifier and draft that name one snapshot.
///
/// That call shape materialises the weights twice for no speedup. It is the
/// shape every sidecar serve branch used to have, and the one a new drafter
/// kind would re-introduce by copy. The rejection is a path check, so it fires
/// before any I/O and needs no snapshot on disk.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: a temp dir this process cannot create is an environment failure, and the panic names it"
)]
fn load_speculative_rejects_one_snapshot_on_both_sides() {
    // An empty directory: reaching `load_model` at all would fail on the
    // missing config.json, so a bare `is_err()` would pass either way. The
    // assertions below name the rejection instead. `TempDir` gives this run
    // its own path and removes it on drop, panic included.
    let tmp = tempfile::tempdir().expect("temp dir");
    let dir = tmp.path();
    let name = dir.file_name().expect("tempdir path has a final component");

    // Two spellings of one directory are still one directory. `Path` equality
    // alone does not see through the second one — canonicalisation does.
    let aliased = dir.join("..").join(name);
    assert_ne!(dir, aliased.as_path());
    for draft in [dir, aliased.as_path()] {
        let msg = SpeculativeDispatcher::load_speculative(dir, draft, Device::Cpu)
            .err()
            .map_or_else(String::new, |e| e.to_string());
        assert!(
            msg.contains("same snapshot directory"),
            "draft={}: expected the same-snapshot rejection, got: {msg:?}",
            draft.display()
        );
    }
}

/// Live spec_forward(K=4) on a single small snapshot. `spec_forward` routes
/// the verifier only, so a verifier-only dispatcher is the right shape.
/// Only checks shape `[1, K, vocab]`.
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn spec_forward_k4_returns_correct_shape() {
    let Some(path_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E2B").map(std::path::PathBuf::from)
    else {
        eprintln!("[spec_test] skipping: RMLX_TEST_MODEL_GEMMA4_E2B not set");
        return;
    };
    let path = path_buf.as_path();
    if !path.exists() {
        eprintln!("[spec_test] snapshot absent — skipping");
        return;
    }

    let disp =
        SpeculativeDispatcher::load_verifier_only(path, Device::Cpu).expect("load_verifier_only");
    // BOS + a few synthetic tokens from gemma vocab range.
    let ids: Vec<u32> = vec![2, 105, 2364, 107, 4368, 105];
    let k = 4_usize;
    let logits = disp.spec_forward(&ids, k).expect("spec_forward");
    let shape = logits.shape();
    assert_eq!(shape.len(), 3, "expected [1,K,vocab], got shape={shape:?}");
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1] as usize, k);
    assert_eq!(shape[2] as usize, disp.vocab_size());
}

// ---------------------------------------------------------------------------
// prefill_chunked exit-sweep invariant
// ---------------------------------------------------------------------------

/// Every cache that entered prefill must run `exit_prefill` before the spec
/// `prefill_chunked` engine returns — on the failure path too.
///
/// This pins the inverse of the shared-helper invariant: an early `?` at the
/// per-chunk forward would strand `caches[i..]` with `in_prefill = true` and no
/// decode seed, so the next decode on a reused cache errors or corrupts KV. The
/// forward is injected here (no live model) and fails on the FIRST chunk, so
/// every cache is one an inline `return Err` would have stranded. Uses
/// `KvQuant::None` caches: no Metal allocation happens because the forward never
/// writes K/V.
#[test]
fn spec_prefill_chunked_runs_exit_sweep_on_failure() {
    let mut caches: Vec<KvCache> = (0..3)
        .map(|_| KvCache::with_quant_max_seq(KvQuant::None, 8))
        .collect();
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    // The closure also proves the assertion below is not vacuous: enter_prefill
    // really set the flag on every cache before the forward runs.
    let forward = |_chunk: &[u32], caches: &mut [KvCache]| -> Result<()> {
        assert!(
            caches.iter().all(KvCache::in_prefill),
            "precondition: prefill_chunked_with must enter prefill on every cache"
        );
        Err(Error::Other(
            "simulated spec first-chunk failure".to_owned(),
        ))
    };
    let _ = prefill_chunked_with(&tokens, &mut caches, 4, Device::Cpu, forward);
    for (i, c) in caches.iter_mut().enumerate() {
        assert!(
            !c.in_prefill(),
            "cache {i} was left in prefill after a failed chunk — the exit_prefill \
             sweep was skipped, so its state is un-finalized for the next decode"
        );
        // Consistent + reusable: a fresh enter/exit cycle succeeds and leaves
        // the cache out of prefill. A stranded (double-swept or un-finalized)
        // cache would not accept a clean re-bracket here.
        c.enter_prefill();
        assert!(
            c.exit_prefill(Device::Cpu).is_ok(),
            "cache {i} must be reusable after a failed prefill sweep"
        );
        assert!(!c.in_prefill(), "cache {i} must exit prefill on reuse");
    }
}

/// The spec prefill engine reports the **first** cause, not a later cascade.
///
/// The forward fails on the first chunk with a distinctive message; the
/// `exit_prefill` sweep for `KvQuant::None` caches is a no-op, so the only error
/// in flight is the forward's. It must reach the caller verbatim.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the regression under test, and its panic names it"
)]
fn spec_prefill_chunked_reports_first_cause() {
    let mut caches: Vec<KvCache> = (0..2)
        .map(|_| KvCache::with_quant_max_seq(KvQuant::None, 8))
        .collect();
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let forward = |_chunk: &[u32], _caches: &mut [KvCache]| -> Result<()> {
        Err(Error::Other("simulated spec prefill cause".to_owned()))
    };
    let err = prefill_chunked_with(&tokens, &mut caches, 4, Device::Cpu, forward)
        .expect_err("a rejected spec prefill must surface as Err, not a silent Ok");
    assert!(
        err.to_string().contains("simulated spec prefill cause"),
        "the underlying cause must reach the caller verbatim, got: {err}"
    );
}

/// The reported decode rate covers the emitted tokens, not the prefill.
///
/// A round loop's total elapsed time starts before the verifier prefill, so
/// dividing the emitted count by it reports a rate that falls as the prompt
/// grows — on a 4k prompt that understated the measured rate by more than
/// half. The window opens at the first emitted token, which is what makes a
/// speculative rate comparable with the ordinary `decode_tps`.
///
/// Instants are injected rather than slept for: a sleep-calibrated bound is a
/// nondeterministic gate under a loaded `cargo test --workspace`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: two marks were just made on the line above, so `tps` returning None would itself be the failure this asserts"
)]
fn decode_window_excludes_the_time_before_the_first_token() {
    let t0 = Instant::now();
    let mut w = DecodeWindow::new();
    // Stand-in for prefill: real time passes before any token is emitted.
    w.mark_at(t0 + Duration::from_millis(120));
    w.mark_at(t0 + Duration::from_millis(160));

    // One 40 ms inter-token gap is exactly 25 tok/s. Counting the 120 ms
    // prefill as well would give 1/0.160 = 6.25.
    let tps = w.tps().expect("two marks span a measurable interval");
    assert!(
        (tps - 25.0).abs() < 1e-9,
        "decode rate {tps} is not the inter-token rate; the window must open at the first emitted token"
    );
}

/// A window with fewer than two marks reports nothing, not zero.
///
/// `0.0` in that slot prints, averages and wins a champion cell exactly like a
/// real throughput of zero, which is the same reason `rmlx baseline` carries
/// its phase timings as `Option`.
#[test]
fn decode_window_reports_none_before_two_tokens() {
    let mut w = DecodeWindow::new();
    assert!(w.tps().is_none(), "an empty window has no rate to report");
    w.mark_at(Instant::now());
    assert!(w.tps().is_none(), "one token spans no interval");
}

/// Every token a round loop emits also advances the window.
///
/// This is the property the reported rate rests on: the window derives its
/// numerator from its own mark count, so the count is only right if
/// `emit_step` is the single door into `emitted`. Dropping the `mark()` from
/// `emit_step`, or pushing to `emitted` around it, shows up here as a
/// mismatch.
///
/// What this does **not** gate: that each round loop logs `window.tps()`
/// rather than recomputing a rate from `elapsed_ms`. That is one line per
/// loop with no server-free oracle. The structural guard there is that `tps`
/// takes no token count, so the old expression cannot be restored by
/// substituting an argument.
#[test]
fn emit_step_advances_the_window_once_per_token() {
    let tk = tiny_tokenizer();
    let mut emitted: Vec<ProbeStep> = Vec::new();
    let mut window = DecodeWindow::new();
    let mut seen = 0_usize;
    let mut step_fn = |_: &ProbeStep| -> Option<u32> {
        seen += 1;
        None
    };

    for id in [1_u32, 2, 3, 4, 5] {
        emit_step(&tk, id, &mut step_fn, &mut emitted, &mut window);
    }

    assert_eq!(emitted.len(), 5, "every call must push a step");
    assert_eq!(
        window.marks(),
        emitted.len(),
        "the window fell behind the emitted buffer — a rate built on its mark count would run fast"
    );
    assert_eq!(seen, 5, "every emitted token must reach the sink");
}

/// A minimal in-memory tokenizer: `emit_step` only calls `id_to_token`.
///
/// Built from a literal vocabulary rather than a snapshot on disk — a helper
/// that returned `None` when the checkout has no models would make the test
/// above skip silently, and a test that skips is not a gate.
#[allow(
    clippy::expect_used,
    reason = "test-only: the vocabulary is the literal three lines above, so a build failure is a broken `tokenizers` dependency and the panic names it"
)]
fn tiny_tokenizer() -> tokenizers::Tokenizer {
    use tokenizers::models::wordlevel::WordLevel;

    let vocab = (0_u32..8).map(|i| (format!("tok{i}"), i)).collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("tok0".to_owned())
        .build()
        .expect("literal vocabulary builds a WordLevel model");
    tokenizers::Tokenizer::new(model)
}
