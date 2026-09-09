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
    // And a block with more than one bonus slot. The refusal is `!=` and not
    // `<`: a verifier block one too long is a verify forward that scored a
    // position the round never proposed for, and reading a correction out of it
    // emits a token from the wrong position.
    assert!(accept_prefix(&[10, 11, 12, 99, 0], &[10, 11, 12], 8).is_err());
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

// ── VerifierDraw ─────────────────────────────────────────────────────────────

/// Build a `[1, k, vocab]` logits array from row-major f32 values.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn logits_block(rows: &[[f32; 4]]) -> Array {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[1, rows.len() as i32, 4], Dtype::F32)
        .expect("a well-formed logits block")
}

fn greedy_cfg() -> crate::sampler::SamplerConfig {
    crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(7),
        top_logprobs_k: 0,
    }
}

fn sampled_cfg() -> crate::sampler::SamplerConfig {
    crate::sampler::SamplerConfig {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.0,
        seed: Some(7),
        top_logprobs_k: 0,
    }
}

/// At temperature 0 the draw is each row's own argmax, and nothing else reaches
/// it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn a_greedy_draw_is_the_argmax_of_every_row() {
    let block = logits_block(&[
        [0.1, 9.0, 0.2, 0.3],
        [4.0, 0.0, 0.5, 8.5],
        [7.0, 1.0, 2.0, 3.0],
    ]);
    let mut draw = VerifierDraw::new(&greedy_cfg());
    assert!(!draw.sampling(), "temperature 0 must not sample");
    let got = draw
        .block_tokens(&block, 3, Device::Cpu)
        .expect("three rows");
    assert_eq!(
        got,
        vec![1, 3, 0],
        "each position takes its own row's argmax"
    );
}

/// The seed a loop emits after prefill is the draw the round loop would take,
/// at a block of one.
///
/// The two seams are separate calls on separate shapes — a `[1, 1, vocab]`
/// prefill row against a `[1, k, vocab]` verified block — so a loop that sampled
/// its block and argmaxed its seed would look right at every position but the
/// first, and one position in a stream is below what a distributional gate can
/// resolve. This is what covers it instead.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn the_seed_draw_is_the_block_draw_at_one_position() {
    for cfg in [greedy_cfg(), sampled_cfg()] {
        let row = logits_block(&[[2.0, 2.1, 1.9, 0.4]]);
        let seed = VerifierDraw::new(&cfg)
            .seed_token(&row, Device::Cpu)
            .expect("one seed token");
        let block = VerifierDraw::new(&cfg)
            .block_tokens(&row, 1, Device::Cpu)
            .expect("one block token");
        assert_eq!(
            vec![seed],
            block,
            "the prefill seam and the round seam must draw the same token from the same \
             row at temperature {}",
            cfg.temperature
        );
    }
}

/// A sampled draw reaches past the argmax, and reproduces itself from its seed.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn a_sampled_draw_is_neither_the_argmax_nor_irreproducible() {
    // Four near-equal logits: id 1 is the argmax at every position, so a block
    // that comes back all ones is an argmax wearing a temperature.
    let rows = [[2.00_f32, 2.05, 2.02, 1.98]; 64];
    let block = logits_block(&rows);
    let cfg = sampled_cfg();
    let first = VerifierDraw::new(&cfg)
        .block_tokens(&block, 64, Device::Cpu)
        .expect("sixty-four rows");
    let again = VerifierDraw::new(&cfg)
        .block_tokens(&block, 64, Device::Cpu)
        .expect("sixty-four rows");
    assert_eq!(first, again, "one seed must give one stream");
    assert!(
        first.iter().any(|&t| t != 1),
        "every position took the argmax, so this is not sampling: {first:?}"
    );
    assert!(
        first
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            >= 3,
        "four near-equal logits must reach more than two ids over sixty-four draws: {first:?}"
    );
}

// -- Round tape: the recurrent rollback's evidence ---------------------------

/// A deterministic ramp, so every position of every taped buffer is distinct
/// and a join that lands one position out is visible in the bytes.
fn ramp(seed: f32, n: usize) -> Vec<f32> {
    (0..n).map(|i| seed + i as f32 * 0.125).collect()
}

#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn tape_arr(seed: f32, shape: &[i32]) -> Array {
    let n: i32 = shape.iter().product();
    Array::from_f32_slice(&ramp(seed, n as usize), shape).expect("from_f32_slice")
}

#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn tape_bytes(a: &Array) -> Vec<u8> {
    a.eval().expect("eval");
    a.to_bytes().expect("to_bytes")
}

/// Shapes the recurrence kernel accepts: `Dk` a multiple of 32, `Hv` a multiple
/// of `Hk`.
const TAPE_HK: i32 = 1;
const TAPE_HV: i32 = 2;
const TAPE_D: i32 = 32;
/// The depthwise conv1d's carried tail, `kernel - 1`.
const TAPE_PAD: i32 = 3;
const TAPE_CONV_DIM: i32 = 2;

/// One taped forward over `len` positions, its arrays cut from a longer ramp
/// starting at `from` so segments of one round line up end to end.
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn tape_segment(from: usize, len: usize) -> GdnTapeSegment {
    let pos = |seed: f32, per: i32| -> Array {
        let per = per as usize;
        let all = ramp(seed, (from + len) * per);
        let shape_len = len as i32;
        let tail = all.get(from * per..).expect("ramp covers the segment");
        Array::from_f32_slice(tail, &[1, shape_len, per as i32]).expect("from_f32_slice")
    };
    let heads = |seed: f32, h: i32| -> Array {
        let flat = pos(seed, h * TAPE_D);
        flat.reshape(&[1, len as i32, h, TAPE_D], Device::Cpu)
            .expect("reshape")
    };
    GdnTapeSegment {
        q: heads(0.5, TAPE_HK),
        k: heads(1.5, TAPE_HK),
        v: heads(2.5, TAPE_HV),
        g: pos(0.25, TAPE_HV),
        beta: pos(0.75, TAPE_HV),
        // The conv input opens with the `kernel - 1` positions carried in from
        // the previous call, so a segment starting at `from` covers the global
        // conv rows `from .. from + pad + len`.
        conv_input: {
            let per = TAPE_CONV_DIM as usize;
            let rows = TAPE_PAD as usize + len;
            let all = ramp(9.0, (from + rows) * per);
            let tail = all.get(from * per..).expect("ramp covers the segment");
            Array::from_f32_slice(tail, &[1, rows as i32, TAPE_CONV_DIM]).expect("from_f32_slice")
        },
        len,
    }
}

/// The whole round as one segment, which is what the refold must reproduce
/// from however many segments actually recorded it.
fn tape_zero_state() -> Array {
    tape_arr(0.0, &[1, TAPE_HV, TAPE_D, TAPE_D])
}

#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn armed(segments: Vec<GdnTapeSegment>, state_in: &Array) -> LinearAttnCache {
    let mut cache = LinearAttnCache::new();
    cache.arm_tape();
    let tape = cache.tape.as_mut().expect("armed");
    for (idx, seg) in segments.into_iter().enumerate() {
        // Only the first forward starts from the round's state. Later forwards
        // start where their predecessor ended, and a tape that kept one of
        // those would refold from the wrong place — so they are handed a state
        // no correct refold can produce.
        let state = if idx == 0 {
            state_in.try_clone().expect("clone round state")
        } else {
            tape_arr(7.0, &[1, TAPE_HV, TAPE_D, TAPE_D])
        };
        tape.push(&state, seg).expect("push");
    }
    cache
}

/// A layer whose tape was never armed cannot be rolled back, and says so rather
/// than leaving the recurrent state where the rejected drafts left it.
///
/// This is the mutation that matters: without the check, the refold has nothing
/// to fold and the obvious implementation leaves the state untouched — a silent
/// no-op that reads as a successful rollback and produces wrong tokens with no
/// error anywhere.
#[test]
fn refold_refuses_a_layer_whose_tape_was_never_armed() {
    let mut lin = vec![armed(vec![tape_segment(0, 3)], &tape_zero_state())];
    // The mutation: this round's arming never reached the layer.
    let _ = lin.get_mut(0).map(LinearAttnCache::take_tape);

    let err = refold_lin_tapes(&mut lin, 3, 2, false, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("recurrent layer 0 has no round tape"),
        "an unarmed layer must be refused by name; got: {err:?}"
    );
}

/// A tape that covers fewer positions than the round fed describes a different
/// round, and refolding it would leave the recurrent state at a prefix the K/V
/// stack beside it does not agree with.
#[test]
fn refold_refuses_a_tape_that_is_short_of_the_round() {
    // The mutation: the round fed five positions across two forwards and only
    // the first was recorded.
    let mut lin = vec![armed(vec![tape_segment(0, 2)], &tape_zero_state())];

    let err = refold_lin_tapes(&mut lin, 5, 3, false, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("taped 2 positions over 1 forwards but the round fed 5"),
        "a short tape must be refused with both counts; got: {err:?}"
    );
}

/// A hybrid hands one recurrent slot per decoder layer, and most of them belong
/// to full-attention layers. Those record nothing and hold no state, and the
/// refold walks past them.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn a_layer_that_holds_no_recurrence_is_walked_past() {
    let state_in = tape_zero_state();
    let mut lin = vec![
        armed(vec![tape_segment(0, 3)], &state_in),
        // A full-attention layer's slot: armed with the rest of the stack, and
        // no forward ever came through it.
        armed(vec![], &state_in),
    ];

    refold_lin_tapes(&mut lin, 3, 0, false, Device::Cpu).expect("refold");

    let unused = lin.get(1).expect("two layers");
    assert!(
        unused.conv_state.is_none() && unused.delta_state.is_none(),
        "a slot that holds no recurrence must be left alone, not given one"
    );
}

/// A layer that DOES hold recurrent state and recorded nothing is the defect
/// that skip must not swallow: its state advanced with the round and there is
/// nothing to roll it back with.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn a_recurrent_layer_that_recorded_nothing_is_still_refused() {
    let state_in = tape_zero_state();
    let mut lin = vec![armed(vec![], &state_in)];
    // The mutation: the round advanced this layer's recurrence and the tape
    // missed every forward that did it.
    lin.get_mut(0).expect("one layer").delta_state = Some(tape_zero_state());

    let err = refold_lin_tapes(&mut lin, 3, 1, false, Device::Cpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("taped 0 positions over 0 forwards but the round fed 3"),
        "a recurrent layer with an empty tape must be refused; got: {err:?}"
    );
}

/// A round that kept nothing goes back to the state it started from, and the
/// conv tail goes back to the one the round carried in — with no kernel call,
/// because there is nothing to fold.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn refold_at_zero_kept_restores_what_the_round_started_from() {
    let state_in = tape_zero_state();
    let seg = tape_segment(0, 3);
    let carried = seq_range(&seg.conv_input, 0, TAPE_PAD, Device::Cpu).expect("carried tail");
    let mut lin = vec![armed(vec![seg], &state_in)];

    refold_lin_tapes(&mut lin, 3, 0, false, Device::Cpu).expect("refold");

    let cache = lin.first().expect("one layer");
    assert_eq!(
        tape_bytes(cache.delta_state.as_ref().expect("delta restored")),
        tape_bytes(&state_in),
        "a zero-kept refold must restore the pre-round recurrent state"
    );
    assert_eq!(
        tape_bytes(cache.conv_state.as_ref().expect("conv restored")),
        tape_bytes(&carried),
        "a zero-kept refold must restore the conv tail the round carried in"
    );
    assert!(
        cache.tape.is_none(),
        "the refold consumes the tape it folded"
    );
}

/// Refolding a one-forward tape at `kept` gives exactly the state a forward
/// over `kept` positions would have left.
///
/// The recurrence is sequential in position, so the state after position `kept`
/// does not depend on how many positions came after it — which is the whole
/// reason the accepted prefix can be rebuilt from what the round already
/// computed. Bytes, not a tolerance: same kernel, same inputs, same order.
#[test]
#[ignore = "requires Metal GPU context (gated_delta recurrence kernel)"]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn a_one_forward_tape_refolds_to_the_shorter_forward() {
    let device = Device::Gpu;
    let round = 5usize;
    let state_in = tape_zero_state();

    for kept in 1..=round {
        let seg = tape_segment(0, round);
        let want = {
            let (_y, state) = crate::gated_delta_msl::gated_delta_step_gpu(
                &seq_range(&seg.q, 0, kept as i32, device).expect("q"),
                &seq_range(&seg.k, 0, kept as i32, device).expect("k"),
                &seq_range(&seg.v, 0, kept as i32, device).expect("v"),
                &seq_range(&seg.g, 0, kept as i32, device).expect("g"),
                &seq_range(&seg.beta, 0, kept as i32, device).expect("beta"),
                &state_in,
                device,
            )
            .expect("reference recurrence");
            state
        };
        let conv_want = seq_range(&seg.conv_input, kept as i32, kept as i32 + TAPE_PAD, device)
            .expect("conv tail");

        let mut lin = vec![armed(vec![tape_segment(0, round)], &state_in)];
        refold_lin_tapes(&mut lin, round, kept, false, device).expect("refold");
        let cache = lin.first().expect("one layer");

        assert_eq!(
            tape_bytes(cache.delta_state.as_ref().expect("delta")),
            tape_bytes(&want),
            "refold at kept={kept} must equal a forward over {kept} positions"
        );
        assert_eq!(
            tape_bytes(cache.conv_state.as_ref().expect("conv")),
            tape_bytes(&conv_want),
            "refold at kept={kept} must leave the conv tail at that position"
        );
    }
}

/// The two-model draft rollback spans a forward per drafted token, so its tape
/// accumulates. Refolding across the join must give the same state a single
/// forward over the same positions would.
///
/// The prefix is walked over every `kept` in the round, so the case that lands
/// exactly on a segment boundary and the ones either side are all covered — a
/// join that took one position too many or too few from a segment moves the
/// state and shows up here.
#[test]
#[ignore = "requires Metal GPU context (gated_delta recurrence kernel)"]
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array this test just built from a slice of its own is present by construction, and the panic names it"
)]
fn an_accumulating_tape_refolds_across_its_segment_join() {
    let device = Device::Gpu;
    let (first, second) = (2usize, 3usize);
    let round = first + second;
    let state_in = tape_zero_state();
    let whole = tape_segment(0, round);

    for kept in 1..=round {
        let want = {
            let (_y, state) = crate::gated_delta_msl::gated_delta_step_gpu(
                &seq_range(&whole.q, 0, kept as i32, device).expect("q"),
                &seq_range(&whole.k, 0, kept as i32, device).expect("k"),
                &seq_range(&whole.v, 0, kept as i32, device).expect("v"),
                &seq_range(&whole.g, 0, kept as i32, device).expect("g"),
                &seq_range(&whole.beta, 0, kept as i32, device).expect("beta"),
                &state_in,
                device,
            )
            .expect("reference recurrence");
            state
        };
        let conv_want = seq_range(
            &whole.conv_input,
            kept as i32,
            kept as i32 + TAPE_PAD,
            device,
        )
        .expect("conv tail");

        let mut lin = vec![armed(
            vec![tape_segment(0, first), tape_segment(first, second)],
            &state_in,
        )];
        refold_lin_tapes(&mut lin, round, kept, false, device).expect("refold");
        let cache = lin.first().expect("one layer");

        assert_eq!(
            tape_bytes(cache.delta_state.as_ref().expect("delta")),
            tape_bytes(&want),
            "a two-segment refold at kept={kept} must equal one forward over {kept} positions"
        );
        assert_eq!(
            tape_bytes(cache.conv_state.as_ref().expect("conv")),
            tape_bytes(&conv_want),
            "a two-segment refold at kept={kept} must leave the conv tail at that position"
        );
    }
}

// -- Round tape against the replay it replaced, on a real model --------------

/// The hybrids the tape is checked on: GDN stacks whose recurrent layers are
/// interleaved with full-attention ones, which is the shape that made the replay
/// run the whole layer stack in the first place. One dense, one mixture — the
/// projections a shorter forward recomputes go through different kernels on the
/// two, and this equality is a claim about those.
const TAPE_REPLAY_SLUGS: &[&str] = &[
    "mlx-community__Qwen3.8-27B-4bit",
    "mlx-community__Qwen3.6-35B-A3B-8bit",
];

/// The snapshots present on this machine, or a named stand-down the GPU runner
/// counts for each that is not.
fn tape_replay_models(test: &str) -> Vec<std::path::PathBuf> {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!("SKIP {test}: RMLX_O_MODELS_ROOT is not set");
        return Vec::new();
    };
    let root = std::path::PathBuf::from(root);
    TAPE_REPLAY_SLUGS
        .iter()
        .filter_map(|slug| {
            let path = root.join(slug);
            if path.join("config.json").exists() {
                return Some(path);
            }
            eprintln!("SKIP {test}: {slug} is not under RMLX_O_MODELS_ROOT");
            None
        })
        .collect()
}

/// A fresh unquantized cache stack, so nothing between the two arms differs but
/// the way the recurrent state got where it is.
#[allow(
    clippy::expect_used,
    reason = "test-only: a cache stack sized from the model just loaded is present by construction, and the panic names it"
)]
fn tape_replay_stack(arch: &Architecture) -> (Vec<KvCache>, Vec<LinearAttnCache>) {
    let n = arch.num_hidden_layers();
    (
        (0..n).map(|_| KvCache::with_quant(KvQuant::None)).collect(),
        (0..n).map(|_| LinearAttnCache::new()).collect(),
    )
}

/// Every recurrent buffer in the stack, read back as f32 in layer order.
#[allow(
    clippy::expect_used,
    reason = "test-only: an Array the model just wrote is present by construction, and the panic names it"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tape_replay_state(lin: &[LinearAttnCache], device: Device) -> Vec<f32> {
    let mut out = Vec::new();
    for cache in lin {
        for buf in [&cache.delta_state, &cache.conv_state]
            .into_iter()
            .flatten()
        {
            let f32_buf = buf.astype(Dtype::F32, device).expect("astype f32");
            f32_buf.eval().expect("materialise");
            out.extend(
                f32_buf
                    .to_bytes()
                    .expect("to_bytes")
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
            );
        }
    }
    out
}

/// `||a - b|| / ||a||`, the relative size of the difference between two
/// states.
///
/// A norm and not a worst element: these buffers hold millions of values, most
/// of them near zero, and an elementwise ratio saturates on any pair of tiny
/// opposite-signed ones whatever the states as a whole are doing.
fn tape_replay_rel_err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "two states of the same stack differ in size"
    );
    let diff: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if scale > 0.0 {
        diff / scale
    } else {
        diff
    }
}

/// Twenty-odd deterministic ids well inside any vocabulary. The values do not
/// matter; that both arms see the same ones does.
const TAPE_REPLAY_PROMPT: &[u32] = &[
    2, 9707, 11, 1879, 13, 576, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 6323, 374, 26867,
    13,
];
/// The round's own tokens: a verify block's worth.
const TAPE_REPLAY_ROUND: &[u32] = &[785, 6722, 315, 15236, 374];

/// Advance a stack over `tokens` the way a round's forwards do, and hand back
/// the recurrent state the round leaves.
///
/// `chunks` is how the round split those tokens between forwards: one chunk for
/// a verify block, one chunk per drafted token for a two-model drafter. An empty
/// chunk is a round that advanced nothing, which is the control at `kept == 1`.
#[allow(
    clippy::expect_used,
    reason = "test-only: a forward over ids this test chose is expected to succeed, and the panic names it"
)]
fn tape_replay_run(
    arch: &Architecture,
    prompt: &[u32],
    chunks: &[&[u32]],
    arm_tapes: bool,
    device: Device,
) -> (Vec<KvCache>, Vec<LinearAttnCache>) {
    let (mut kv, mut lin) = tape_replay_stack(arch);
    arch.forward_seq_last_k_with_cache(prompt, 1, &mut kv, Some(&mut lin), device)
        .expect("prefill");
    if arm_tapes {
        arm_lin_tapes(Some(&mut lin));
    }
    for chunk in chunks.iter().filter(|c| !c.is_empty()) {
        arch.forward_seq_last_k_with_cache(chunk, 1, &mut kv, Some(&mut lin), device)
            .expect("round forward");
    }
    (kv, lin)
}

/// The tape rebuilds the state the replay used to rebuild, on a real recurrent
/// stack — for a round taken as one verify forward, and for one taken as a
/// forward per token.
///
/// The replay arm is what this change removed: restore the pre-round state and
/// run the accepted prefix through the whole layer stack. It is reproduced here
/// as a second stack prefilled identically and advanced over the accepted prefix
/// alone, which is the same computation.
///
/// **How close they can be is the model's own answer, and the run measures it.**
/// The replay computes the prefix in a forward of its own length; the tape
/// returns what the round's forward computed at those positions. On the dense
/// hybrid the two are bit-identical, and the run says so. On the mixture the
/// model does not reproduce itself that way — one forward over the round and the
/// same tokens stepped one at a time part company by a couple of percent, which
/// is a property of that stack and not of this change — so the bound the refold
/// is held to is that same disagreement, measured beside it in the same process.
///
/// The control is what gives the comparison power: the same refold against the
/// replay one token short of the accepted length, which is where a refold that
/// folded the wrong number of positions would sit. It reads about 0.5 against a
/// refold-to-replay agreement of at most a few percent.
#[test]
#[ignore = "requires Metal GPU context and a 27B snapshot"]
#[allow(
    clippy::expect_used,
    reason = "test-only: a model this test has already checked for existence but cannot load is a broken checkout, and the panic names it"
)]
#[allow(
    clippy::print_stderr,
    reason = "test-only: stand-down notice and the measured agreement, both read by the operator"
)]
fn a_round_tape_refolds_to_what_the_replay_produced() {
    const NAME: &str = "a_round_tape_refolds_to_what_the_replay_produced";
    let device = Device::Gpu;
    let round = TAPE_REPLAY_ROUND;
    let per_token: Vec<&[u32]> = round.iter().map(std::slice::from_ref).collect();

    for path in tape_replay_models(NAME) {
        let model = path.file_name().unwrap_or_default().to_string_lossy();
        let arch = load_model(&path, device, &LoadOpts::default()).expect("load verifier");
        assert!(
            arch.needs_lin_caches(),
            "{model} must carry recurrent state or this test proves nothing"
        );

        // What this model's own two regimes make of the same tokens: the round
        // in one forward against the round stepped. It is the bound the refold
        // is held to, because no rebuild of a prefix can be closer to a forward
        // over that prefix than the model is to itself.
        let (_kv, batched) = tape_replay_run(&arch, TAPE_REPLAY_PROMPT, &[round], false, device);
        let (_kv, stepped) = tape_replay_run(&arch, TAPE_REPLAY_PROMPT, &per_token, false, device);
        let regime = tape_replay_rel_err(
            &tape_replay_state(&batched, device),
            &tape_replay_state(&stepped, device),
        );
        eprintln!("[{NAME}/{model}] one forward against stepped: {regime:.6}");

        for (shape, chunks) in [
            ("one verify forward", vec![round]),
            ("a forward per token", per_token.clone()),
        ] {
            for kept in 1..round.len() {
                let (_kv, mut taped) =
                    tape_replay_run(&arch, TAPE_REPLAY_PROMPT, &chunks, true, device);
                refold_lin_tapes(&mut taped, round.len(), kept, false, device).expect("refold");
                let refolded = tape_replay_state(&taped, device);

                let (_kv, replayed) = tape_replay_run(
                    &arch,
                    TAPE_REPLAY_PROMPT,
                    &[round.get(..kept).expect("kept prefix")],
                    false,
                    device,
                );
                let agreement =
                    tape_replay_rel_err(&refolded, &tape_replay_state(&replayed, device));
                let (_kv, off_by_one) = tape_replay_run(
                    &arch,
                    TAPE_REPLAY_PROMPT,
                    &[round.get(..kept - 1).expect("shorter prefix")],
                    false,
                    device,
                );
                let control =
                    tape_replay_rel_err(&refolded, &tape_replay_state(&off_by_one, device));

                eprintln!(
                    "[{NAME}/{shape}] kept={kept} agreement={agreement:.6} \
                     one-token-short={control:.6}"
                );
                assert!(
                    agreement <= regime,
                    "{model}, {shape} at kept={kept}: the refold and the replay of the \
                     same {kept} tokens differ by {agreement}, and this model \
                     reproduces itself to {regime}"
                );
                assert!(
                    agreement * 10.0 < control,
                    "{model}, {shape} at kept={kept}: the refold is no nearer the replay \
                     of {kept} tokens ({agreement}) than the replay of {} ({control}), \
                     so this cell would pass whatever the refold folded",
                    kept - 1
                );
            }
        }
    }
}

/// A two-model round drafts no more than one verify forward can score beside
/// the carry token.
#[test]
fn the_two_model_draft_count_stops_at_the_verify_ceiling() {
    assert_eq!(two_model_drafts_per_round(4), 4);
    assert_eq!(
        two_model_drafts_per_round(MAX_BLOCK_SIZE - 1),
        MAX_BLOCK_SIZE - 1
    );
    assert_eq!(
        two_model_drafts_per_round(MAX_BLOCK_SIZE),
        MAX_BLOCK_SIZE - 1
    );
    assert_eq!(two_model_drafts_per_round(usize::MAX), MAX_BLOCK_SIZE - 1);
}

// ── block_capped_by_checkpoint ───────────────────────────────────────────────

/// The request wins when it asks for less than the checkpoint was trained at.
#[test]
fn a_request_narrower_than_the_checkpoint_runs_at_the_request() {
    assert_eq!(block_capped_by_checkpoint(5, 8), 5);
}

/// The checkpoint wins when the request asks for more than it was trained at:
/// the selector's chain is defined over the trained block and no wider.
#[test]
fn a_request_wider_than_the_checkpoint_runs_at_the_checkpoint() {
    assert_eq!(block_capped_by_checkpoint(8, 5), 5);
}

/// Two positions is the floor at both ends — a block of one is the seed alone
/// and drafts nothing, so a request or a checkpoint below it still runs a round
/// that proposes something.
#[test]
fn neither_side_can_take_the_block_below_a_seed_and_one_draft() {
    assert_eq!(block_capped_by_checkpoint(0, 8), 2);
    assert_eq!(block_capped_by_checkpoint(8, 1), 2);
    assert_eq!(block_capped_by_checkpoint(0, 0), 2);
}

/// A checkpoint whose config never went through `check_config` is still bounded.
///
/// This is the case the clamp exists for, and the only one that separates it
/// from the loader's refusal: `DFlash2Drafter` is publicly
/// constructible with public fields, so `declared` here is whatever the caller
/// put in the struct. Without the clamp this returns that number, and the round
/// sizes its token buffer, its verify input and its selector chain from it.
#[test]
fn a_config_the_loader_never_saw_is_still_bounded_by_one_verify_forward() {
    assert_eq!(
        block_capped_by_checkpoint(usize::MAX, usize::MAX),
        MAX_BLOCK_SIZE
    );
    assert_eq!(
        block_capped_by_checkpoint(usize::MAX, 4_294_967_295),
        MAX_BLOCK_SIZE
    );
    // And the request alone cannot lift it past the checkpoint either.
    assert_eq!(block_capped_by_checkpoint(usize::MAX, 8), 8);
}

/// The ceiling admits its own value and refuses the next one, so it cannot be
/// off by one in either direction.
#[test]
fn the_ceiling_admits_itself_and_nothing_above() {
    assert_eq!(
        block_capped_by_checkpoint(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE),
        MAX_BLOCK_SIZE
    );
    assert_eq!(
        block_capped_by_checkpoint(MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE + 1),
        MAX_BLOCK_SIZE
    );
    assert_eq!(
        block_capped_by_checkpoint(MAX_BLOCK_SIZE - 1, MAX_BLOCK_SIZE),
        MAX_BLOCK_SIZE - 1
    );
}
