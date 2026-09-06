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

/// Every registered architecture class maps to its own prefill-chunk key.
///
/// The chunk is keyed on this mapping, so a class routed to another
/// architecture's key prefills at a size no sweep ever measured for it. Keys
/// are asserted rather than the chunks they resolve to: most classes share a
/// chunk value with some other class, so a value-only comparison stays green
/// through exactly the misrouting this pins — `Qwen3ForCausalLM` to `gemma4`
/// is invisible while their defaults agree.
///
/// The table is compared against the registry as a set, so a newly registered
/// architecture fails here rather than being skipped.
#[test]
fn verifier_prefill_chunk_is_the_architectures_own() {
    // `JinaEmbeddingsV4Model` is an encoder with no `Architecture` variant and
    // so no verifier; the empty key is the conservative fallback chunk, which
    // is the right answer for a class that has no prefill path of its own.
    let expected: &[(&str, &str)] = &[
        ("Gemma4ForConditionalGeneration", "gemma4"),
        ("Gemma4UnifiedForConditionalGeneration", "gemma4"),
        ("Gemma3ForConditionalGeneration", "gemma3"),
        ("Qwen2ForCausalLM", "qwen2"),
        ("Qwen3ForCausalLM", "qwen3"),
        ("LagunaForCausalLM", "laguna"),
        ("Qwen3_5MoeForConditionalGeneration", "qwen3_5_moe"),
        ("Qwen3_5ForConditionalGeneration", "qwen3_5_moe"),
        ("Qwen3VLMoeForConditionalGeneration", "qwen3_vl_moe"),
        ("BitNetForCausalLM", "bitnet"),
        ("JinaEmbeddingsV4Model", ""),
    ];

    let mut covered: Vec<&str> = expected.iter().map(|(class, _)| *class).collect();
    covered.sort_unstable();
    let mut registered: Vec<&str> = crate::arch::registry::KNOWN_ARCHS.to_vec();
    registered.sort_unstable();
    assert_eq!(
        covered, registered,
        "this table and the architecture registry describe different sets"
    );

    for (class, key) in expected {
        assert_eq!(
            crate::prefill_chunk::module_key_for_class(class),
            *key,
            "{class} does not resolve to the {key} prefill-chunk key"
        );
    }
}

/// The chunk sizes the verifier prefill actually cuts the prompt into are the
/// architecture's own.
///
/// The test above pins the lookup; this one pins the wiring, by observing the
/// slices `prefill_chunked_for_class` hands its forward. Without it the lookup
/// could be correct and unused — the call site is one argument, and an
/// argument that stopped naming the architecture would fail no assertion.
#[test]
fn verifier_prefill_cuts_the_prompt_at_the_architectures_chunk() {
    if std::env::var("RMLX_PREFILL_CHUNK").is_ok() {
        return;
    }

    for class in ["Qwen3ForCausalLM", "Gemma3ForConditionalGeneration"] {
        let key = crate::prefill_chunk::module_key_for_class(class);
        // A per-arch override outranks the arch default, so under one the
        // source assertion below would fail on a correct resolver.
        if std::env::var(format!("RMLX_PREFILL_CHUNK_{}", key.to_uppercase())).is_ok() {
            continue;
        }
        // Against the shipped constant, not against a second call to the
        // resolver: comparing the resolver to itself would hold whatever it
        // returned.
        let chunk = crate::prefill_chunk::arch_default(key).unwrap_or(64);
        let tokens: Vec<u32> = (0..(chunk * 2 + 3) as u32).collect();
        let mut seen: Vec<usize> = Vec::new();
        let result =
            prefill_chunked_for_class(class, &tokens, &mut [], Device::Cpu, |slice, _caches| {
                seen.push(slice.len());
                Ok(())
            });
        assert!(result.is_ok(), "{class}: {result:?}");
        assert_eq!(
            seen,
            vec![chunk, chunk, 3],
            "{class} prefill did not cut the prompt at its own chunk"
        );
    }
}

/// Deterministic `[1, 1, s, 2]` f32 K/V pair for the rollback tests below.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rollback_kv(s: i32, base: f32) -> Array {
    let mut data: Vec<f32> = Vec::with_capacity((s * 2) as usize);
    for p in 0..s {
        data.push(base + p as f32);
        data.push(base + p as f32 + 0.5);
    }
    Array::from_f32_slice(&data, &[1, 1, s, 2]).unwrap()
}

/// A prompt then three decode steps, which leaves a windowed layer rotated and
/// a plain one able to roll back to anywhere.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn filled_layer(window: Option<i32>, device: Device) -> KvCache {
    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::None, 512, window);
    cache
        .update(&rollback_kv(6, 0.0), &rollback_kv(6, 100.0), device)
        .unwrap();
    for step in 0..3 {
        let p = (6 + step) as f32;
        cache
            .update(&rollback_kv(1, p), &rollback_kv(1, 100.0 + p), device)
            .unwrap();
    }
    cache
}

/// `truncate_kv_to` moves every layer or none.
///
/// A stack left half rolled back is the same desync the ring fix exists to
/// stop, reached through the failure path instead of through a silent no-op:
/// the layers that did move sit behind an offset the refusing one still holds,
/// and no caller can put them back. So the refusal is decided before any layer
/// is touched, and it names the layer that decided it.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "the stack is built with two layers immediately above"
)]
#[allow(
    clippy::panic,
    reason = "the else-branch of a let-else that only a broken gate can reach"
)]
fn a_stack_with_one_layer_that_cannot_roll_back_moves_no_layer() {
    let device = Device::Cpu;
    // Layer 0 could reach the target on its own. Layer 1 is a window that
    // decode writes left rotated, and cannot.
    let mut stack = vec![filled_layer(None, device), filled_layer(Some(4), device)];
    assert_eq!((stack[0].offset(), stack[1].offset()), (9, 9));
    assert!(stack[0].can_truncate_to(8));
    assert!(!stack[1].can_truncate_to(8));

    let Err(err) = truncate_kv_to(&mut stack, 8) else {
        panic!("a stack holding a layer that cannot reach 8 must not report success")
    };
    assert_eq!(
        (stack[0].offset(), stack[1].offset()),
        (9, 9),
        "the layer that could roll back must not have"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("layer 1"),
        "the refusal must name the layer: {msg}"
    );
    assert!(
        msg.contains("cannot be rolled back to 8"),
        "and the target it could not reach: {msg}"
    );

    // And the gate does not stand in the way of a rollback the whole stack can
    // make. A target both layers are already at would prove nothing — it is
    // `roll_back(0)` on one and a no-op on the other, and a loop that skipped
    // the truncation outright would still pass it. So this one moves.
    let mut reachable = vec![filled_layer(None, device), filled_layer(None, device)];
    assert!(truncate_kv_to(&mut reachable, 7).is_ok());
    assert_eq!(
        (reachable[0].offset(), reachable[1].offset()),
        (7, 7),
        "a target every layer can reach must move every layer"
    );
}

// ---------------------------------------------------------------------------
// Vocabulary pairing
// ---------------------------------------------------------------------------

/// A literal vocabulary: `pieces[i]` is the piece at id `i`.
fn vocab_of(pieces: &[&str]) -> HashMap<String, u32> {
    pieces
        .iter()
        .enumerate()
        .map(|(id, piece)| ((*piece).to_owned(), id as u32))
        .collect()
}

/// The pairs `load_speculative` admits: one vocabulary spelled twice, and one
/// that differs only by a short tail of specials the other side never emits —
/// the shape a base and an audio release of one family actually ship.
#[test]
fn vocab_verdict_admits_identical_and_short_tail_pairs() {
    let base = vocab_of(&["<pad>", "<bos>", "a", "b", "c"]);
    assert!(vocab_pairing_verdict(&base, &base).is_ok());

    let mut with_tail = base.clone();
    for i in 0..7_u32 {
        with_tail.insert(format!("<|special_{i}|>"), 5 + i);
    }
    assert!(vocab_pairing_verdict(&base, &with_tail).is_ok());
    assert!(
        vocab_pairing_verdict(&with_tail, &base).is_ok(),
        "the tolerance is symmetric — either side may carry the tail"
    );
}

/// A pair that agrees on size and disagrees on meaning is the case a
/// `vocab_size` comparison cannot see, and the one that serves garbage rather
/// than failing. The refusal names the id and both pieces.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the defect under test"
)]
fn vocab_verdict_refuses_a_piece_that_differs_and_names_it() {
    let verifier = vocab_of(&["<pad>", "<bos>", "a", "b", "c"]);
    let draft = vocab_of(&["<pad>", "<bos>", "a", "B", "c"]);
    let msg = vocab_pairing_verdict(&verifier, &draft)
        .expect_err("same size, different piece at id 3")
        .to_string();
    assert!(msg.contains("token id 3"), "names the id: {msg}");
    assert!(
        msg.contains("\"b\"") && msg.contains("\"B\""),
        "names both pieces: {msg}"
    );

    // An id one side skips inside the shared range is a difference too, not a
    // tail: the other side can propose it.
    let mut holed = verifier.clone();
    holed.remove("a");
    let msg = vocab_pairing_verdict(&verifier, &holed)
        .expect_err("a hole inside the shared range")
        .to_string();
    assert!(
        msg.contains("token id 2") && msg.contains("absent"),
        "{msg}"
    );
}

/// A tail past the tolerance is a different vocabulary, however well the
/// prefix agrees. The bound is the one llama.cpp admits, so the two engines
/// accept the same pairs.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the defect under test"
)]
fn vocab_verdict_refuses_a_tail_past_the_tolerance() {
    let base = vocab_of(&["<pad>", "<bos>", "a"]);
    let mut long_tail = base.clone();
    for i in 0..=(VOCAB_TAIL_TOLERANCE as u32) {
        long_tail.insert(format!("<|extra_{i}|>"), 3 + i);
    }
    let msg = vocab_pairing_verdict(&base, &long_tail)
        .expect_err("a tail of tolerance + 1 ids")
        .to_string();
    assert!(
        msg.contains(&format!("{} more ids", VOCAB_TAIL_TOLERANCE + 1)),
        "names the tail size: {msg}"
    );

    long_tail.remove(&format!("<|extra_{VOCAB_TAIL_TOLERANCE}|>"));
    assert!(
        vocab_pairing_verdict(&base, &long_tail).is_ok(),
        "exactly the tolerance is admitted"
    );
}

/// An id two pieces claim has no single meaning, and letting one win by hash
/// order would make the verdict irreproducible. Refused naming the id and both
/// pieces, on either side.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the defect under test"
)]
fn vocab_verdict_refuses_an_id_two_pieces_claim() {
    let clean = vocab_of(&["<pad>", "<bos>", "a", "b"]);
    let mut doubled = clean.clone();
    doubled.insert("B".to_owned(), 3);
    for (verifier, draft, side) in [(&clean, &doubled, "draft"), (&doubled, &clean, "verifier")] {
        let msg = vocab_pairing_verdict(verifier, draft)
            .expect_err("two pieces at id 3")
            .to_string();
        assert!(
            msg.contains(side) && msg.contains("token id 3 twice"),
            "{msg}"
        );
        assert!(
            msg.contains("\"b\"") && msg.contains("\"B\""),
            "names both pieces: {msg}"
        );
    }
}

/// A vocabulary reaching past the ceiling is refused by name rather than
/// walked, so one sentinel at a huge id cannot turn model load into a spin.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the defect under test"
)]
fn vocab_verdict_refuses_an_id_past_the_ceiling() {
    let mut huge = vocab_of(&["<pad>", "<bos>", "a"]);
    huge.insert("<sentinel>".to_owned(), VOCAB_ID_CEILING);
    let msg = vocab_pairing_verdict(&huge, &huge)
        .expect_err("a shared id at the ceiling")
        .to_string();
    assert!(
        msg.contains(&format!("token id {VOCAB_ID_CEILING}")),
        "names the id: {msg}"
    );
}

/// `load_speculative` runs the verdict before it reads a config or a weight.
///
/// Two snapshot directories holding nothing but a `tokenizer.json` each: with
/// the gate in place the refusal names the differing id; without it the call
/// fails later, on the missing `config.json`, and says nothing about tokens.
/// Every other test drives the verdict directly, so this is the one that fails
/// when the call is deleted.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the tokenizers are literal and the tempdir is this process's own, so a failure to write either names a broken environment"
)]
fn load_speculative_refuses_a_foreign_tokenizer_before_reading_weights() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let write = |name: &str, pieces: &[&str]| -> std::path::PathBuf {
        use tokenizers::models::wordlevel::WordLevel;
        let dir = tmp.path().join(name);
        std::fs::create_dir(&dir).expect("snapshot dir");
        let vocab = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| ((*piece).to_owned(), id as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_owned())
            .build()
            .expect("literal vocabulary builds a WordLevel model");
        tokenizers::Tokenizer::new(model)
            .save(dir.join("tokenizer.json"), false)
            .expect("write tokenizer.json");
        dir
    };
    let verifier = write("verifier", &["<unk>", "<bos>", "sea", "sky"]);
    let draft = write("draft", &["<unk>", "<bos>", "sea", "SKY"]);

    let msg = SpeculativeDispatcher::load_speculative(&verifier, &draft, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        msg.contains("token id 3"),
        "the pair must be refused on the token, before any weight is read: {msg:?}"
    );
    assert!(
        !msg.contains("config.json"),
        "the refusal reached the config read — the vocabulary gate did not run: {msg:?}"
    );
}

// ---------------------------------------------------------------------------
// Acceptance walk
// ---------------------------------------------------------------------------

/// The walk, unwrapped, for the cases whose block is well formed.
#[allow(
    clippy::expect_used,
    reason = "test-only: each caller passes a block one longer than its proposals, which is the shape the walk accepts"
)]
fn walk(verifier: &[u32], draft: &[u32], budget: usize) -> (usize, Vec<u32>) {
    accept_prefix(verifier, draft, budget).expect("a block one longer than its proposals")
}

#[test]
fn accept_prefix_all_accepted_emits_the_bonus_token() {
    let (acc, emit) = walk(&[10, 11, 12, 99], &[10, 11, 12], 8);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11, 12, 99]);
}

#[test]
fn accept_prefix_stops_at_the_first_disagreement_and_emits_the_correction() {
    let (acc, emit) = walk(&[10, 11, 55, 0], &[10, 11, 12], 8);
    assert_eq!(acc, 2);
    assert_eq!(emit, vec![10, 11, 55]);
}

#[test]
fn accept_prefix_emits_only_the_correction_when_nothing_is_accepted() {
    let (acc, emit) = walk(&[42, 0, 0], &[10, 11], 8);
    assert_eq!(acc, 0);
    assert_eq!(emit, vec![42]);
}

#[test]
fn accept_prefix_budget_caps_the_emission_and_not_the_acceptance() {
    // The round committed three drafts to the caches whatever the token budget
    // was; a walk that reported two accepts here would leave the KV holding a
    // position the loop believes it rolled back.
    let (acc, emit) = walk(&[10, 11, 12, 99], &[10, 11, 12], 2);
    assert_eq!(acc, 3);
    assert_eq!(emit, vec![10, 11]);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the call is deliberately malformed and an Ok here is the assertion failing"
)]
fn accept_prefix_refuses_a_block_that_is_not_its_proposals_plus_a_bonus() {
    // The two arguments are same-typed slices and the order carries the whole
    // meaning. Swapped, this compiles and — before the check — returned an
    // accept count that then drove the KV rollback.
    let verifier = [10u32, 11, 12, 99];
    let draft = [10u32, 11, 12];
    let err = accept_prefix(&draft, &verifier, 8).expect_err("arguments the wrong way round");
    let msg = err.to_string();
    assert!(
        msg.contains('3') && msg.contains('4'),
        "the refusal must name both counts so a swapped call site is identifiable, got: {msg}"
    );
    // A block missing its bonus slot is the same defect arriving from the other
    // side: there is no correction to emit and nothing should guess one.
    assert!(accept_prefix(&[10, 11], &[10, 11], 8).is_err());
}

// ---------------------------------------------------------------------------
// Reading the verifier's argmax back
// ---------------------------------------------------------------------------

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the fixture buffer is built two lines above with the byte count the call asks for, so an Err here is the assertion failing"
)]
fn argmax_tokens_reads_one_id_per_verified_position() {
    let bytes: Vec<u8> = [7u32, 9, 11].iter().flat_map(|v| v.to_le_bytes()).collect();
    let got = argmax_tokens(&bytes, 3).expect("three positions, twelve bytes");
    assert_eq!(got, vec![7, 9, 11]);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the fixture buffer is built two lines above with the byte count the call asks for, so an Err here is the assertion failing"
)]
fn argmax_tokens_stops_at_the_block_and_ignores_a_longer_buffer() {
    let bytes: Vec<u8> = [7u32, 9, 11].iter().flat_map(|v| v.to_le_bytes()).collect();
    let got = argmax_tokens(&bytes, 2).expect("two positions asked for");
    assert_eq!(got, vec![7, 9]);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the fixture buffer is built two lines above with the byte count the call asks for, so an Err here is the assertion failing"
)]
fn argmax_tokens_names_a_short_buffer_instead_of_panicking() {
    // The read runs once per round. A slice-and-unwrap here aborts the request
    // with a bounds panic and no mention of the device that came back short.
    let bytes: Vec<u8> = [7u32, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
    let err = argmax_tokens(&bytes, 3).expect_err("three positions, eight bytes");
    let msg = err.to_string();
    assert!(
        msg.contains('8') && msg.contains("12") && msg.contains('3'),
        "the refusal must name the bytes it got, the bytes it needed and the \
         positions it was reading for, got: {msg}"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: the fixture buffer is built two lines above with the byte count the call asks for, so an Err here is the assertion failing"
)]
fn argmax_tokens_of_an_empty_block_reads_nothing() {
    assert_eq!(
        argmax_tokens(&[], 0).expect("no positions"),
        Vec::<u32>::new()
    );
}

// --- unread_tensor_refusal ---

/// A snapshot every tensor of which was read loads; one carrying tensors the
/// loader has no code for is refused, naming them and naming the loader that
/// refused.
///
/// Both directions, because the quiet direction is what keeps the check from
/// decaying into noise a reader learns to skip. Mutation this fails on:
/// `!consumed.contains(name)` -> `consumed.contains(name)`, which refuses the
/// supported checkpoint and admits the unsupported one.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions: panicking on unexpected values is intentional"
)]
fn a_snapshot_a_loader_only_half_reads_is_refused_and_a_whole_one_is_not() {
    use std::collections::HashSet;

    let read_by_the_loader = ["fc.weight", "hidden_norm.weight", "norm.weight"];
    let consumed: HashSet<String> = read_by_the_loader.iter().map(|s| (*s).to_owned()).collect();

    let whole: HashSet<String> = read_by_the_loader.iter().map(|s| (*s).to_owned()).collect();
    assert!(
        unread_tensor_refusal("DFlashDrafter", &whole, &consumed).is_ok(),
        "a snapshot the loader reads entirely must load"
    );

    // A checkpoint generation newer than the loader: weight families it has no
    // code for.
    let mut partial = whole;
    partial.insert("candidate_selector.successor_codebook".to_owned());
    partial.insert("layers.0.attention_conv.base_kernel".to_owned());
    let err = unread_tensor_refusal("DFlashDrafter", &partial, &consumed).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("DFlashDrafter"),
        "the refusal must name the loader that issued it: {msg}"
    );
    assert!(
        msg.contains('2'),
        "refusal must count the unread tensors: {msg}"
    );
    assert!(
        msg.contains("candidate_selector.successor_codebook")
            && msg.contains("layers.0.attention_conv.base_kernel"),
        "refusal must name the unread tensors: {msg}"
    );
    assert!(
        !msg.contains("fc.weight"),
        "a consumed tensor must not be reported unread: {msg}"
    );
}
