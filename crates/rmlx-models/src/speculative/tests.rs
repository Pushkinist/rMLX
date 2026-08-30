//! Speculative dispatcher unit tests.

use super::*;

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
