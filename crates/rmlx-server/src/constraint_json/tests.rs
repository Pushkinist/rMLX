use super::*;

fn feed_chars(g: &mut JsonGrammar, s: &str) -> Result<(), ()> {
    for &b in s.as_bytes() {
        g.step(b)?;
    }
    Ok(())
}

#[test]
fn accepts_simple_object() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "{\"a\":1}").expect("valid simple object");
    assert!(g.is_done());
}

#[test]
fn accepts_nested() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "{\"a\":{\"b\":[1,2,3]},\"c\":true}").expect("valid nested");
    assert!(g.is_done());
}

#[test]
fn accepts_top_level_array() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "[1, 2, 3]").expect("valid array");
    assert!(g.is_done());
}

#[test]
fn accepts_top_level_literals() {
    for s in ["true", "false", "null", "42", "-3.14", "\"hello\""] {
        let mut g = JsonGrammar::new();
        feed_chars(&mut g, s).expect("valid literal");
        assert!(g.is_done(), "expected Done after `{s}`");
    }
}

#[test]
fn rejects_unbalanced_brace() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "{\"a\":1").expect("partial parse ok");
    assert!(!g.is_done(), "must not be done with unclosed brace");
}

#[test]
fn rejects_trailing_comma() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "{\"a\":1,").expect("partial parse ok");
    assert!(g.step(b'}').is_err(), "trailing comma should reject `}}`");
}

#[test]
fn rejects_unclosed_string() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "\"abc").expect("partial string ok");
    assert!(!g.is_done());
}

#[test]
fn rejects_bad_literal() {
    // `tru` then `e` is fine, but `tru` then `x` must error.
    let mut g = JsonGrammar::new();
    g.step(b't').unwrap();
    g.step(b'r').unwrap();
    g.step(b'u').unwrap();
    assert!(g.step(b'x').is_err(), "`trux` must reject");
}

#[test]
fn property_accepts_generated_json() {
    // Property test: feed each known-valid string into the grammar AND
    // through serde_json — both must accept.
    // Seed: deterministic list, no PRNG.
    let cases: &[&str] = &[
        "{}",
        "[]",
        "null",
        "true",
        "false",
        "0",
        "123",
        "-0",
        "-1.5e10",
        "\"\"",
        "\"hello world\"",
        "{\"k\":\"v\"}",
        "[1,2,3,4,5]",
        "[true,false,null]",
        "{\"a\":1,\"b\":2,\"c\":3}",
        "{\"a\":[1,{\"b\":2}],\"c\":\"d\"}",
        "[[[]]]",
        "{\"esc\":\"a\\nb\\tc\"}",
        "{\"u\":\"\\u00e9\"}",
        "{\"a\":1.0e-5}",
        "{ \"a\" :  1 , \"b\" : 2 }",
        "[ \"x\" , \"y\" ]",
        "{\"k\":[]}",
        "{\"k\":{}}",
        "[null,true,false,0,\"\",[],{}]",
    ];
    for s in cases {
        let mut g = JsonGrammar::new();
        feed_chars(&mut g, s).unwrap_or_else(|()| panic!("rejected valid JSON: {s}"));
        serde_json::from_str::<serde_json::Value>(s)
            .unwrap_or_else(|e| panic!("serde rejected control case `{s}`: {e}"));
        assert!(g.is_done(), "grammar not Done after `{s}`");
    }
}

#[test]
fn property_rejects_invalid() {
    // Cases the grammar should reject either at a byte step or by
    // refusing to reach Done. (`1.2.3` is a known-loose case for
    // the number sub-state — the masker tolerates a second `.` to
    // keep state-machine complexity manageable. serde_json would
    // ultimately reject, but the masker won't.)
    let bad: &[&str] = &[
        "{",
        "{,}",
        "[1,]",
        "{\"a\":}",
        "{\"a\":1,}",
        "\"unterm",
        "tru",
        "fals",
        "nul",
        "[1 2]",
        "{a:1}",
        "{\"a\" 1}",
    ];
    for s in bad {
        let mut g = JsonGrammar::new();
        let res = feed_chars(&mut g, s);
        assert!(
            res.is_err() || !g.is_done(),
            "grammar accepted invalid input as complete: {s}"
        );
    }
}

// ── synthetic-vocab tests for the mask layer ────────────────────────────

/// Build a TokenBytesMap from a list of (token_bytes) — id is index.
fn synthetic_bytes_map(entries: &[&[u8]]) -> Arc<TokenBytesMap> {
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u32> = vec![0];
    for e in entries {
        bytes.extend_from_slice(e);
        offsets.push(bytes.len() as u32);
    }
    Arc::new(TokenBytesMap {
        bytes,
        offsets,
        vocab_size: entries.len(),
    })
}

/// Returns a Qwen3 tokenizer if the snapshot exists locally; tests that
/// need a real tokenizer use this and skip on absence.
fn maybe_tokenizer() -> Option<Arc<tokenizers::Tokenizer>> {
    let dir = std::env::var_os("RMLX_TEST_MODEL_QWEN36").map(std::path::PathBuf::from)?;
    let p = dir.join("tokenizer.json");
    if !p.exists() {
        return None;
    }
    let tk = tokenizers::Tokenizer::from_file(&p).ok()?;
    Some(Arc::new(tk))
}

#[test]
fn token_bytes_map_qwen_smoke() {
    let Some(tk) = maybe_tokenizer() else {
        eprintln!("[json_constraint] tokenizer absent, skipping");
        return;
    };
    let map = TokenBytesMap::new(&tk);
    // Some token must decode starting with `{`.
    let any_brace_start =
        (0..map.vocab_size()).any(|id| map.token_bytes(id).first() == Some(&b'{'));
    assert!(any_brace_start, "no token decodes to `{{` first byte");
}

#[test]
fn mask_at_start_only_value_starters() {
    // Synthetic vocab: 0={ 1=} 2=" 3=a 4=: 5=1 6=eos(empty)
    let pm = synthetic_bytes_map(&[b"{", b"}", b"\"", b"a", b":", b"1", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![6]);
    c.force_engage();
    let m = c.step_mask(7);
    assert!(m[0], "`{{` must be allowed at Start");
    assert!(m[2], "`\"` must be allowed at Start");
    assert!(m[5], "digit `1` must be allowed at Start");
    assert!(!m[1], "`}}` must NOT be allowed at Start");
    assert!(!m[3], "`a` must NOT be allowed at Start");
    assert!(!m[4], "`:` must NOT be allowed at Start");
    assert!(!m[6], "EOS (empty) must NOT be allowed mid-grammar");
    c.feed_bytes(b"{");
    let m2 = c.step_mask(7);
    assert!(m2[2], "after `{{`, `\"` (key start) must be allowed");
    assert!(m2[1], "after `{{`, `}}` (empty obj) must be allowed");
}

#[test]
fn bound_round_trip_synthetic() {
    let pm = synthetic_bytes_map(&[b"{", b"}", b"\"", b"a", b":", b"1", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![6]);
    c.force_engage();
    let script: &[(usize, &[u8])] = &[
        (0, b"{"),
        (2, b"\""),
        (3, b"a"),
        (2, b"\""),
        (4, b":"),
        (5, b"1"),
        (1, b"}"),
    ];
    let mut out: Vec<u8> = Vec::new();
    for (id, bytes) in script {
        let m = c.step_mask(7);
        assert!(m[*id], "token {id} must be allowed at this step");
        c.feed_bytes(bytes);
        out.extend_from_slice(bytes);
    }
    let s = std::str::from_utf8(&out).unwrap();
    assert!(c.finished(), "constraint must reach Done after `{s}`");
    serde_json::from_str::<serde_json::Value>(s).expect("must parse");
    let m_end = c.step_mask(7);
    assert!(m_end[6], "EOS must be allowed at terminal state");
}

#[test]
fn mask_rejects_multibyte_token_with_illegal_continuation() {
    // Vocab where token id 0 is the multi-byte ` Apple` (space + Apple).
    // The space alone would pass at Start (whitespace), but `A` after
    // it must fail. The per-step mask should therefore mark id 0
    // FORBIDDEN even though its first byte (space) is allowed.
    let pm = synthetic_bytes_map(&[b" Apple", b"{", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![2]);
    c.force_engage();
    let m = c.step_mask(3);
    assert!(!m[0], "multi-byte ` Apple` must be rejected at Start");
    assert!(m[1], "`{{` must be allowed at Start");
}

#[test]
fn warm_up_passes_then_engages_on_value_starter() {
    // Vocab: 0=filler (`AAAA`), 1=`{`, 2=eos. With `is_thinking=false`
    // (default), engagement fires on the first `{`.
    let pm = synthetic_bytes_map(&[b"AAAA", b"{", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![2]);
    // Pre-engagement: mask is all-true.
    let m_warm = c.step_mask(3);
    assert!(m_warm.iter().all(|&b| b), "warm-up mask must be all-true");
    // Feed filler — no `{`, stay in warm-up.
    c.advance(0);
    let m_still_warm = c.step_mask(3);
    assert!(
        m_still_warm.iter().all(|&b| b),
        "still warm-up after non-`{{` token"
    );
    // Feed `{` — engages.
    c.advance(1);
    let m_strict = c.step_mask(3);
    assert!(!m_strict[0], "engaged mask must reject non-value token");
}

#[test]
fn warm_up_blocks_engagement_during_thinking() {
    let pm = synthetic_bytes_map(&[b"{", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![1]);
    // Signal that the model is currently thinking. The engagement
    // scan must skip the `{` byte in this token.
    c.is_thinking.store(true, Ordering::Relaxed);
    c.advance(0);
    assert!(!c.engaged, "engagement must be deferred while thinking");
    // Now thinking ends — next `{` engages.
    c.is_thinking.store(false, Ordering::Relaxed);
    c.advance(0);
    assert!(c.engaged, "engagement must fire after thinking ends");
}

// ── A6.5: fence suppression tests ───────────────────────────────────────

/// Pre-engagement text that is only ` ```json\n ` must be flagged as a
/// fence so the handler knows to discard it.
#[test]
fn fence_suppression_pure_fence() {
    let pm = synthetic_bytes_map(&[b"```json\n", b"{", b"}", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![3]);
    // Token 0 decodes to ` ```json\n` — pre-engagement, fence-only.
    c.advance(0);
    assert!(!c.engaged, "fence token must not trigger engagement");
    assert!(
        c.pre_engage_is_fence(),
        "pure fence pre-text must be detected"
    );
    // Token 1 is `{` — engagement fires.
    c.advance(1);
    assert!(c.engaged, "engagement must fire on `{{` token");
}

/// Pre-engagement text that contains real prose must NOT be silently
/// discarded (it is not a fence).
#[test]
fn fence_suppression_real_prose_not_discarded() {
    let pm = synthetic_bytes_map(&[b"Sure, here: ", b"{", b"}", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![3]);
    c.advance(0); // real prose pre-text
    assert!(!c.engaged);
    // pre_engage_is_fence must be false for real prose.
    assert!(
        !c.pre_engage_is_fence(),
        "real prose pre-text must NOT be flagged as fence"
    );
}

/// Object/array root: engagement still requires `{`/`[` (not broken by A6.5).
#[test]
fn object_root_engage_policy_unaffected() {
    let pm = synthetic_bytes_map(&[b"AAAA", b"{", b"}", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(pm, vec![3]);
    // Non-structural token → no engagement.
    c.advance(0);
    assert!(!c.engaged, "no engagement on non-structural pre-text");
    // `{` token → engagement.
    c.advance(1);
    assert!(c.engaged, "engagement on `{{`");
}

#[test]
fn diagnose_gemma_token_bytes() {
    let Some(dir) = std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
    else {
        eprintln!("[diagnose_gemma] RMLX_TEST_MODEL_GEMMA4_E4B not set, skipping");
        return;
    };
    let p = dir.join("tokenizer.json");
    if !p.exists() {
        eprintln!("[diagnose_gemma] tokenizer absent, skipping");
        return;
    }
    let tok = tokenizers::Tokenizer::from_file(&p).unwrap();
    let map = TokenBytesMap::new(&tok);

    // Find all tokens whose decoded bytes start with { or [
    let mut starters: Vec<(usize, Vec<u8>)> = Vec::new();
    for id in 0..map.vocab_size() {
        let b = map.token_bytes(id);
        if b.first() == Some(&b'{') || b.first() == Some(&b'[') {
            starters.push((id, b.to_vec()));
        }
    }
    eprintln!("Tokens starting with {{/[ : {} found", starters.len());
    for (id, bytes) in starters.iter().take(20) {
        eprintln!(
            "  id={} len={} bytes={:?}",
            id,
            bytes.len(),
            &bytes[..bytes.len().min(8)]
        );
    }

    // Check if any multi-byte starters would fail the grammar after {
    let mut grammar_after_brace = JsonGrammar::new();
    grammar_after_brace.step(b'{').unwrap();
    let allowed_after_brace = grammar_after_brace.allowed_bytes();

    let bad_starters: Vec<_> = starters
        .iter()
        .filter(|(_, bytes)| bytes.len() > 1 && !allowed_after_brace[bytes[1] as usize])
        .collect();
    eprintln!(
        "Bad starters (invalid byte after {{/[): {} found",
        bad_starters.len()
    );
    for (id, bytes) in bad_starters.iter().take(10) {
        eprintln!("  id={} bytes[1]={} ({:?})", id, bytes[1], bytes[1] as char);
    }
    // Check what low-ID tokens decode to (special tokens like <pad>, <unused0>...)
    eprintln!("Low-ID token bytes (potential special tokens):");
    for id in [0usize, 1, 2, 3, 4, 5, 6, 7, 8] {
        let b = map.token_bytes(id);
        eprintln!(
            "  id={} len={} bytes={:?} utf8={:?}",
            id,
            b.len(),
            &b[..b.len().min(8)],
            std::str::from_utf8(b).unwrap_or("<invalid>")
        );
    }

    // Check how many tokens in the mask would be allowed at Start state
    let g_start = JsonGrammar::new();
    let allowed_at_start = g_start.allowed_bytes();
    let start_allowed_count = (0..map.vocab_size())
        .filter(|&id| {
            let b = map.token_bytes(id);
            !b.is_empty() && allowed_at_start[b[0] as usize]
        })
        .count();
    eprintln!("Tokens whose first byte is allowed at Start: {start_allowed_count}");

    // Check if any token whose first byte is NOT allowed at Start has empty bytes
    let empty_count = (0..map.vocab_size())
        .filter(|&id| map.token_bytes(id).is_empty())
        .count();
    eprintln!("Tokens that decode to empty bytes: {empty_count}");

    // This test is diagnostic — it must NOT fail even if bad_starters is non-empty.
    // The Gemma vocab legitimately has multi-byte tokens whose second byte is
    // non-printable; the engagement logic must handle them gracefully.
    eprintln!("[diagnose_gemma] done");
}

// ───────────────────────────────────────────────────────────────────────────
// Empirical probe: clone-per-token vs scratch-reuse cost in step_mask.
//
// Reproduces the production step_mask inner loop (mod.rs) three ways and times
// each across grammar states of increasing nesting depth, on a synthetic
// ~152K-token vocab with a BPE-like byte-length distribution. Answers:
//   (b) ~ms per decode step on the constrained path,
//   (c) what fraction is per-token clone allocation vs irreducible step work,
//   (d) whether eliminating the alloc (scratch reuse) is worth it, or whether
//       the cost is algorithmic (vocab-wide probe → DFA/trie index).
//
// Run: cargo test -p rmlx-server constraint_json::tests::probe_step_mask \
//        --profile release-perf -- --ignored --nocapture
// ───────────────────────────────────────────────────────────────────────────

/// Deterministic LCG — no PRNG dep, reproducible vocab.
struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Build a realistic ~`n`-token vocab: structural JSON tokens up front, then
/// BPE-like text tokens (mean ~4-5 bytes, some space/quote prefixed).
#[allow(
    clippy::indexing_slicing,
    reason = "letter index is bounded by rng.below(letters.len())"
)]
fn realistic_vocab(n: usize) -> Arc<TokenBytesMap> {
    let structural: &[&[u8]] = &[
        b"{",
        b"}",
        b"[",
        b"]",
        b"\"",
        b":",
        b",",
        b" ",
        b"\n",
        b"\t",
        b"0",
        b"1",
        b"2",
        b"3",
        b"4",
        b"5",
        b"6",
        b"7",
        b"8",
        b"9",
        b"-",
        b".",
        b"true",
        b"false",
        b"null",
        b"\"a\"",
        b"\"name\"",
        b"\"id\"",
        b" \"x",
        b"\"value\"",
    ];
    let mut owned: Vec<Vec<u8>> = Vec::with_capacity(n);
    for s in structural {
        owned.push(s.to_vec());
    }
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let letters = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    while owned.len() < n {
        // length weighted 1..12, mean ~4-5
        let len = match rng.below(100) {
            0..=14 => 1,
            15..=39 => 2,
            40..=64 => 4,
            65..=84 => 6,
            85..=94 => 9,
            _ => 12,
        };
        let mut t: Vec<u8> = Vec::with_capacity(len + 1);
        match rng.below(100) {
            0..=19 => t.push(b' '),  // Ġ-style space prefix
            20..=24 => t.push(b'"'), // quote-prefixed
            _ => {}
        }
        for _ in 0..len {
            let idx = rng.below(letters.len() as u64) as usize;
            t.push(letters[idx]);
        }
        owned.push(t);
    }
    owned.truncate(n);
    let refs: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
    synthetic_bytes_map(&refs)
}

/// Drive the grammar to the state reached after consuming `prefix`.
#[allow(
    clippy::expect_used,
    reason = "prefixes are static valid-JSON fragments authored in this test"
)]
fn grammar_at(prefix: &str) -> JsonGrammar {
    let mut g = JsonGrammar::new();
    for &b in prefix.as_bytes() {
        g.step(b).expect("prefix must be valid JSON-so-far");
    }
    g
}

#[test]
#[ignore = "perf probe — run manually with --ignored --nocapture"]
fn probe_step_mask_clone_vs_scratch() {
    let vocab_n = 152_064usize;
    let bm = realistic_vocab(vocab_n);
    let n = bm.vocab_size();
    let reps = 30u32;

    // (label, json-prefix). Depth = open-frame count after the prefix.
    let scenes: &[(&str, &str)] = &[
        ("Start top-value (depth0)", ""),
        ("ObjectExpectKey (depth1)", "{"),
        ("ExpectValue (depth1)", "{\"a\":"),
        ("Nested ExpectValue (depth3)", "{\"a\":{\"b\":{\"c\":"),
        (
            "Nested ExpectValue (depth8)",
            "{\"a\":[{\"b\":[{\"c\":[{\"d\":[",
        ),
        ("InString value (depth1)", "{\"a\":\""),
    ];

    println!("\nvocab={n}  reps={reps}  (figures are per-decode-step = total/reps)\n");
    println!(
        "{:<30} {:>10} {:>10} {:>7} {:>11} {:>9}",
        "grammar state", "clone_ms", "scrch_ms", "speedup", "allocOnly", "legal"
    );
    println!("{}", "-".repeat(82));

    for (label, prefix) in scenes {
        let g = grammar_at(prefix);
        let mut mask = vec![false; n];

        // A) clone-per-token — current production logic (mod.rs:796).
        let t0 = std::time::Instant::now();
        let mut legal = 0usize;
        for _ in 0..reps {
            mask.fill(false);
            for (id, m) in mask.iter_mut().enumerate() {
                let bytes = bm.token_bytes(id);
                if bytes.is_empty() {
                    continue;
                }
                let mut gg = g.clone();
                let mut ok = true;
                for &b in bytes {
                    if gg.step(b).is_err() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    *m = true;
                }
            }
            legal = mask.iter().filter(|&&m| m).count();
        }
        let clone_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        let mask_a = mask.clone();

        // B) scratch-reuse — one grammar, refilled via clear()+extend (no alloc
        //    after warm-up). state is Copy; stack reuses its heap buffer.
        let t1 = std::time::Instant::now();
        let mut scratch = JsonGrammar::new();
        for _ in 0..reps {
            mask.fill(false);
            for (id, m) in mask.iter_mut().enumerate() {
                let bytes = bm.token_bytes(id);
                if bytes.is_empty() {
                    continue;
                }
                scratch.state = g.state;
                scratch.stack.clear();
                scratch.stack.extend_from_slice(&g.stack);
                let mut ok = true;
                for &b in bytes {
                    if scratch.step(b).is_err() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    *m = true;
                }
            }
        }
        let scratch_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        // Correctness: scratch-reuse must produce a byte-identical allow-mask.
        assert_eq!(
            mask, mask_a,
            "scratch mask diverges from clone mask at {label}"
        );

        // C) alloc-only — isolate the clone() heap cost (no stepping).
        let t2 = std::time::Instant::now();
        for _ in 0..reps {
            for id in 0..n {
                if bm.token_bytes(id).is_empty() {
                    continue;
                }
                let gg = g.clone();
                std::hint::black_box(&gg);
            }
        }
        let alloc_ms = t2.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

        let speedup = clone_ms / scratch_ms.max(1e-9);
        println!(
            "{label:<30} {clone_ms:>10.3} {scratch_ms:>10.3} {speedup:>7.2} {alloc_ms:>11.3} {legal:>9}"
        );
    }
    println!();
}

// ── insignificant-whitespace bound (json_object engine) ─────────────────────

/// Same rule as the schema engine: an unbounded run of insignificant
/// whitespace is a cycle a greedy decoder never leaves, so the run is capped.
#[test]
fn json_object_insignificant_whitespace_run_is_bounded() {
    let mut g = JsonGrammar::new();
    feed_chars(&mut g, "{").expect("object opens");

    let cap = MAX_INSIGNIFICANT_WS_RUN as usize;
    let probe = cap * 8 + 64;
    let mut accepted = 0usize;
    while accepted < probe && g.step(b' ').is_ok() {
        accepted += 1;
    }
    assert_eq!(
        accepted, cap,
        "grammar must refuse the whitespace byte after {cap} in a row; it accepted {accepted}"
    );
    // Real progress resets the run — a pretty-printed document still parses.
    feed_chars(&mut g, "\"a\": 1,\n  \"b\": 2\n}").expect("document completes");
    assert!(g.is_done());
}

/// Raw C0 control bytes are illegal inside a JSON string. Accepting them both
/// mis-parses the string and hands a greedy decoder an unbounded run of raw
/// whitespace *inside* a value, which no whitespace bound outside strings can
/// stop.
#[test]
fn json_object_raw_control_byte_inside_a_string_is_rejected() {
    for ctrl in *b"\n\t\r\x01" {
        let mut g = JsonGrammar::new();
        feed_chars(&mut g, "{\"a\":\"x").expect("string value opens");
        assert!(
            g.step(ctrl).is_err(),
            "raw control byte {ctrl:#04x} must be illegal inside a JSON string"
        );
    }
}

/// The decode-loop reproduction for the json_object engine: a decoder that
/// always prefers whitespace must run out, and EOS must stay masked while the
/// object is incomplete.
#[test]
fn json_object_mask_stops_offering_whitespace_once_the_run_is_capped() {
    // ids: 0=`{`  1=`\n`  2=`  `  3=`"`  4=EOS
    let bm = synthetic_bytes_map(&[b"{", b"\n", b"  ", b"\"", b""]);
    let mut c = JsonObjectConstraint::from_bytes_map(bm, vec![4]);

    let _ = c.step_mask(5);
    c.advance(0);

    let budget = MAX_INSIGNIFICANT_WS_RUN as usize * 8 + 64;
    let mut ws_tokens = 0usize;
    let mut last = (false, false);
    while ws_tokens < budget {
        let m = c.step_mask(5);
        last = (m[1], m[2]);
        let pick = if last.0 {
            1
        } else if last.1 {
            2
        } else {
            break;
        };
        c.advance(pick);
        ws_tokens += 1;
    }
    assert!(
        ws_tokens < budget,
        "greedy whitespace decoder never ran out: emitted {ws_tokens} whitespace tokens"
    );
    assert_eq!(
        last,
        (false, false),
        "mask must stop offering both whitespace pieces"
    );
    let m = c.step_mask(5);
    assert!(m[3], "the key opener must still be allowed");
    assert!(!m[4], "EOS must stay masked — the object is not complete");
}
