use super::*;

// ── TokenizerKind classification ─────────────────────────────────────

#[test]
fn classify_bytelevel() {
    let v = serde_json::json!({ "decoder": { "type": "ByteLevel" } });
    assert_eq!(
        TokenizerKind::from_tokenizer_json(&v),
        TokenizerKind::ByteLevel
    );
}

#[test]
fn classify_spm_no_strip_gemma() {
    // Exactly Gemma3/Gemma4's decoder (probed from the real
    // tokenizer.json).
    let v = serde_json::json!({
        "decoder": {
            "type": "Sequence",
            "decoders": [
                { "type": "Replace", "pattern": { "String": "\u{2581}" }, "content": " " },
                { "type": "ByteFallback" },
                { "type": "Fuse" }
            ]
        }
    });
    assert_eq!(
        TokenizerKind::from_tokenizer_json(&v),
        TokenizerKind::SpmNoStrip
    );
    assert!(!TokenizerKind::SpmNoStrip.trim_first_space());
}

#[test]
fn classify_spm_strip() {
    let v = serde_json::json!({
        "decoder": {
            "type": "Sequence",
            "decoders": [
                { "type": "Replace", "pattern": { "String": "\u{2581}" }, "content": " " },
                { "type": "ByteFallback" },
                { "type": "Fuse" },
                { "type": "Strip", "content": " ", "start": 1, "stop": 0 }
            ]
        }
    });
    assert_eq!(
        TokenizerKind::from_tokenizer_json(&v),
        TokenizerKind::SpmStrip
    );
    assert!(TokenizerKind::SpmStrip.trim_first_space());
}

#[test]
fn classify_other_when_no_decoder() {
    let v = serde_json::json!({ "model": {} });
    assert_eq!(TokenizerKind::from_tokenizer_json(&v), TokenizerKind::Other);
}

// ── Byte-buffering core (no tokenizer; drive diff_emit directly) ──────
//
// These exercise the healing logic by simulating the sequence of
// `decode(all_ids)` results the HF crate would return: a growing string
// whose tail is `�` whenever the last codepoint is split. This is the
// exact failure shape the repro produced.

/// Drive `diff_emit` with a hand-rolled sequence of full-decode strings
/// and assert the concatenation of segments + a final flush equals the
/// fully-decoded string with no `�`.
fn drive(kind: TokenizerKind, full_decodes: &[&str], final_full: &str) -> String {
    let mut dt = StreamingDetokenizer::new(kind);
    let mut out = String::new();
    for f in full_decodes {
        out.push_str(&dt.diff_emit((*f).to_owned()));
    }
    // Simulate finalize: emit tail past `decoded` for the true full.
    let tail = dt.tail_after_decoded(final_full);
    out.push_str(&dt.maybe_trim_first(tail));
    out
}

#[test]
fn split_at_every_byte_boundary_roundtrips() {
    // "café 🎉 日本語" — Latin-1 accent, 4-byte emoji, 3-byte CJK.
    let s = "café 🎉 日本語";
    let bytes = s.as_bytes();
    for split in 0..=bytes.len() {
        // Two "tokens": bytes[..split] and bytes[split..]. The HF crate
        // would `from_utf8_lossy` each *cumulative* prefix. Simulate the
        // two cumulative full-decodes.
        let first = String::from_utf8_lossy(&bytes[..split]).into_owned();
        let whole = s.to_owned(); // cumulative after 2nd token = exact
        let got = drive(TokenizerKind::ByteLevel, &[&first, &whole], &whole);
        assert_eq!(
            got, s,
            "split at byte {split}: got {got:?}, first-decode was {first:?}"
        );
        assert!(!got.contains('\u{FFFD}'), "U+FFFD leaked at split {split}");
    }
}

#[test]
fn ascii_invariance_identical_to_naive() {
    // Pure ASCII: every byte is its own boundary → no withholding ever.
    // Segments must equal the naive per-step byte-diff exactly.
    let steps = ["H", "He", "Hel", "Hell", "Hello", "Hello ", "Hello w"];
    let mut dt = StreamingDetokenizer::new(TokenizerKind::ByteLevel);
    let mut naive_prev = String::new();
    for f in steps {
        let seg = dt.diff_emit(f.to_owned());
        let naive_seg = f[naive_prev.len()..].to_owned();
        naive_prev = f.to_owned();
        assert_eq!(seg, naive_seg, "ASCII step {f:?} diverged from naive");
    }
}

#[test]
fn fuzz_random_utf8_random_chunking_roundtrips() {
    // Deterministic xorshift64* PRNG — fixed seed, printed on failure.
    const SEED: u64 = 0xA10_F0FD_5EED_1234;
    let mut state = SEED;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545F4914F6CDD1D)
    };

    // Codepoint pools spanning 1..=4 UTF-8 byte lengths.
    let ascii: Vec<char> = ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
    let latin1: Vec<char> = "àáâäçèéêëìíîïñòóôöùúûüÿ".chars().collect();
    let cjk: Vec<char> = "日本語中文字符測試漢字繁體".chars().collect();
    let emoji: Vec<char> = "🎉🎊🥳🚀🔥✨💡🌍🍕🎸".chars().collect();

    for iter in 0..1000 {
        // Build a random valid UTF-8 string, length 1..=40 codepoints.
        let len = 1 + (next() % 40) as usize;
        let mut s = String::new();
        for _ in 0..len {
            let pool = match next() % 4 {
                0 => &ascii,
                1 => &latin1,
                2 => &cjk,
                _ => &emoji,
            };
            let c = pool[(next() as usize) % pool.len()];
            s.push(c);
        }
        let bytes = s.as_bytes();

        // Random token boundaries: 1..=6 cut points at random BYTE
        // offsets (deliberately mid-codepoint allowed). Build the
        // cumulative `from_utf8_lossy` full-decodes the HF crate would
        // produce after each token.
        let n_cuts = 1 + (next() % 6) as usize;
        let mut cuts: Vec<usize> = (0..n_cuts)
            .map(|_| (next() as usize) % (bytes.len() + 1))
            .collect();
        cuts.push(bytes.len());
        cuts.sort_unstable();

        let mut dt = StreamingDetokenizer::new(TokenizerKind::ByteLevel);
        let mut out = String::new();
        let mut last_full = String::new();
        for &cut in &cuts {
            let full = String::from_utf8_lossy(&bytes[..cut]).into_owned();
            out.push_str(&dt.diff_emit(full.clone()));
            last_full = full;
        }
        // finalize: the true full decode is the exact string.
        let tail = dt.tail_after_decoded(&s);
        out.push_str(&tail);

        assert_eq!(
            out, s,
            "fuzz iter {iter} (SEED={SEED:#x}) mismatch:\n  src ={s:?}\n  got ={out:?}\n  cuts={cuts:?}\n  last_full={last_full:?}"
        );
        assert!(
            !out.contains('\u{FFFD}'),
            "fuzz iter {iter} (SEED={SEED:#x}) leaked U+FFFD: {out:?}"
        );
    }
    eprintln!("fuzz_random_utf8_random_chunking_roundtrips: 1000 iters OK, SEED={SEED:#x}");
}

#[test]
fn spm_strip_first_segment_leading_space_only() {
    // Strict-SPM: first segment " The" → "The"; later " world" kept.
    let mut dt = StreamingDetokenizer::new(TokenizerKind::SpmStrip);
    assert_eq!(dt.diff_emit(" The".to_owned()), "The");
    assert_eq!(dt.diff_emit(" The world".to_owned()), " world");
}

#[test]
fn spm_no_strip_keeps_genuine_leading_space() {
    // Gemma (`SpmNoStrip`): a genuine leading space is preserved.
    let mut dt = StreamingDetokenizer::new(TokenizerKind::SpmNoStrip);
    assert_eq!(dt.diff_emit(" The".to_owned()), " The");
}

#[test]
fn bytelevel_never_strips() {
    // Qwen (`ByteLevel`): no leading-space rule, ever.
    let mut dt = StreamingDetokenizer::new(TokenizerKind::ByteLevel);
    assert_eq!(dt.diff_emit(" hello".to_owned()), " hello");
}

#[test]
fn finalize_flushes_truncated_tail_lossy() {
    // Generation genuinely stops mid-codepoint: finalize must emit the
    // lossy tail (a `�`) rather than swallow it.
    let mut dt = StreamingDetokenizer::new(TokenizerKind::ByteLevel);
    // Mid-stream: "ab" then a split codepoint → withheld.
    assert_eq!(dt.diff_emit("ab".to_owned()), "ab");
    assert_eq!(dt.diff_emit("ab\u{FFFD}".to_owned()), "");
    // True EOS with the codepoint still incomplete: lossy flush.
    let tail = dt.tail_after_decoded("ab\u{FFFD}");
    assert_eq!(tail, "\u{FFFD}");
}
