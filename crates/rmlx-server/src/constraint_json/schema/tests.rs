use super::*;
use serde_json::json;

fn node(s: &Value, strict: bool) -> SchemaNode {
    SchemaNode::parse(s, strict).expect("schema parses")
}

fn feed(g: &mut SchemaGrammar, s: &str) -> Result<(), ()> {
    for &b in s.as_bytes() {
        g.step(b)?;
    }
    Ok(())
}

#[test]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
fn schema_parse_shapes() {
    // object
    let o = node(
        &json!({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}),
        false,
    );
    match o {
        SchemaNode::Object {
            props,
            required,
            additional,
        } => {
            assert_eq!(props.len(), 1);
            assert_eq!(required.as_ref(), ["a".to_string()].as_slice());
            assert!(
                additional,
                "non-strict, no additionalProperties → permissive"
            );
        }
        _ => panic!("expected Object"),
    }
    // array
    let a = node(
        &json!({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3}),
        false,
    );
    match a {
        SchemaNode::Array { items, min, max } => {
            assert_eq!(*items, SchemaNode::Num { integer: true });
            assert_eq!(min, Some(2));
            assert_eq!(max, Some(3));
        }
        _ => panic!("expected Array"),
    }
    // enum
    let e = node(&json!({"type":"string","enum":["a","b"]}), false);
    assert_eq!(
        e,
        SchemaNode::Str {
            enum_: Some(vec!["a".into(), "b".into()])
        }
    );
    // oneOf
    let u = node(&json!({"oneOf":[{"const":"a"},{"const":"bb"}]}), false);
    match u {
        SchemaNode::Union(b) => assert_eq!(b.len(), 2),
        _ => panic!("expected Union"),
    }
    // strict tightens
    let s = node(
        &json!({"type":"object","properties":{"x":{"type":"string"},"y":{"type":"integer"}}}),
        true,
    );
    match s {
        SchemaNode::Object {
            required,
            additional,
            ..
        } => {
            assert_eq!(required.len(), 2, "strict ⇒ all props required");
            assert!(!additional, "strict ⇒ additionalProperties:false");
        }
        _ => panic!("expected Object"),
    }
}

#[test]
fn object_required_keys() {
    let n = node(
        &json!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"],"additionalProperties":false}),
        false,
    );
    let mut g = SchemaGrammar::new(n.clone());
    feed(&mut g, "{\"location\":\"Paris\"}").expect("conforming object accepted");
    assert!(g.is_done());

    // Missing required key → `}` rejected right after `{`.
    let mut g2 = SchemaGrammar::new(n);
    g2.step(b'{').unwrap();
    assert!(g2.step(b'}').is_err(), "empty object misses required key");
}

#[test]
fn enum_forces_literal() {
    let n = node(
        &json!({"type":"string","enum":["celsius","fahrenheit"]}),
        false,
    );
    let mut g = SchemaGrammar::new(n.clone());
    feed(&mut g, "\"celsius\"").expect("celsius accepted");
    assert!(g.is_done());

    let mut g2 = SchemaGrammar::new(n);
    g2.step(b'"').unwrap();
    // 'c'/'f' allowed, 'k' (kelvin) blocked.
    let a = g2.allowed_bytes();
    assert!(a[b'c' as usize] && a[b'f' as usize]);
    assert!(!a[b'k' as usize], "non-enum first letter blocked");
    assert!(g2.step(b'k').is_err());
}

#[test]
fn integer_vs_number() {
    let i = node(&json!({"type":"integer"}), false);
    let mut g = SchemaGrammar::new(i);
    g.step(b'4').unwrap();
    assert!(g.step(b'.').is_err(), "integer must reject `.`");

    let f = node(&json!({"type":"number"}), false);
    let mut g2 = SchemaGrammar::new(f);
    feed(&mut g2, "4.2").expect("number accepts 4.2");
    assert!(g2.is_done());
}

#[test]
fn array_bounds() {
    let n = node(
        &json!({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3}),
        false,
    );
    // [1] under min → `]` rejected after one element
    let mut g = SchemaGrammar::new(n.clone());
    feed(&mut g, "[1").unwrap();
    assert!(g.step(b']').is_err(), "below minItems must reject `]`");

    // [1,2] accepted
    let mut g2 = SchemaGrammar::new(n.clone());
    feed(&mut g2, "[1,2]").expect("min satisfied");
    assert!(g2.is_done());

    // [1,2,3,4] over max → 4th element start rejected
    let mut g3 = SchemaGrammar::new(n);
    feed(&mut g3, "[1,2,3").unwrap();
    // after 3 elements at max, `,` must be rejected
    assert!(g3.step(b',').is_err(), "above maxItems must reject `,`");
}

#[test]
fn oneof_discriminated() {
    let n = node(&json!({"oneOf":[{"const":"a"},{"const":"bb"}]}), false);
    let mut g = SchemaGrammar::new(n.clone());
    feed(&mut g, "\"a\"").expect("const a accepted");
    assert!(g.is_done());

    let mut g2 = SchemaGrammar::new(n.clone());
    feed(&mut g2, "\"bb\"").expect("const bb accepted");
    assert!(g2.is_done());

    let mut g3 = SchemaGrammar::new(n);
    g3.step(b'"').unwrap();
    assert!(g3.step(b'c').is_err(), "non-member `c` rejected");
}

#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn round_trip_with_tokenizer() {
    let Some(dir) = std::env::var_os("RMLX_TEST_MODEL_QWEN36").map(std::path::PathBuf::from) else {
        eprintln!("[schema] RMLX_TEST_MODEL_QWEN36 not set, skipping round-trip");
        return;
    };
    let p = dir.join("tokenizer.json");
    if !p.exists() {
        eprintln!("[schema] tokenizer absent, skipping round-trip");
        return;
    }
    let tk = Arc::new(tokenizers::Tokenizer::from_file(&p).unwrap());
    let schema = json!({
        "type":"object",
        "properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}},
        "required":["location","unit"],
        "additionalProperties":false
    });
    let mut c = SchemaConstraint::new(tk.clone(), vec![], &schema, true, None).unwrap();
    c.force_engage();
    // Drive a decode loop greedily: at each step pick the first allowed
    // token whose bytes keep the grammar alive, preferring the target.
    let target = b"{\"location\":\"Paris\",\"unit\":\"celsius\"}";
    let mut out: Vec<u8> = Vec::new();
    let mut guard = 0;
    while !c.finished() && guard < 200 {
        guard += 1;
        let m = c.step_mask(tk.get_vocab_size(true)).to_vec();
        // find a single-byte token equal to the next target byte
        let want = target.get(out.len()).copied();
        let mut picked: Option<u32> = None;
        for (id, &allowed) in m.iter().enumerate().take(tk.get_vocab_size(true)) {
            if !allowed {
                continue;
            }
            let b = c.bytes_map.token_bytes(id);
            if b.len() == 1 && Some(b[0]) == want {
                picked = Some(id as u32);
                break;
            }
        }
        let Some(id) = picked else { break };
        out.extend_from_slice(c.bytes_map.token_bytes(id as usize));
        c.advance(id);
    }
    let s = String::from_utf8(out).unwrap();
    let v: Value = serde_json::from_str(&s).expect("valid JSON");
    assert_eq!(v["location"], "Paris");
    assert_eq!(v["unit"], "celsius");
    assert!(c.finished());
}

#[test]
fn malformed_schema_errors() {
    assert!(SchemaNode::parse(&json!("not an object"), false).is_err());
    assert!(SchemaNode::parse(&json!({"enum":[]}), false).is_err());
    assert!(SchemaNode::parse(&json!({"oneOf":"x"}), false).is_err());
    assert!(SchemaNode::parse(&json!({"type":"object","properties":"bad"}), false).is_err());
}

#[test]
fn unsupported_keyword_graceful_non_strict() {
    // `pattern` on a string → still a Str node, not an error (non-strict).
    let n = node(&json!({"type":"string","pattern":"^x"}), false);
    assert_eq!(n, SchemaNode::Str { enum_: None });
    // dangling `$ref` → UnresolvableRef error (no `$defs` to resolve).
    let r = SchemaNode::parse(&json!({"$ref":"#/$defs/X"}), false);
    assert!(matches!(r, Err(SchemaError::UnresolvableRef(_))));
}

// ── strict-mode 400-on-unsupported-keyword ────────────────────────

#[test]
fn strict_rejects_unsupported_keywords() {
    // Each of these must be a hard error in strict mode and degrade to
    // a valid (Any / bare-type) node in non-strict mode.
    let cases: &[Value] = &[
        json!({"type":"string","pattern":"^[a-z]+$"}),
        json!({"type":"string","format":"email"}),
        json!({"type":"integer","minimum":0}),
        json!({"type":"integer","maximum":10}),
        json!({"type":"string","minLength":1}),
        json!({"type":"string","maxLength":5}),
        json!({"allOf":[{"type":"object"},{"type":"object"}]}),
        json!({"not":{"type":"string"}}),
        json!({"if":{"type":"string"},"then":{"const":"x"}}),
        json!({"type":"object","unevaluatedProperties":false}),
        json!({"type":"array","prefixItems":[{"type":"integer"}]}),
    ];
    for s in cases {
        let strict = SchemaNode::parse(s, true);
        assert!(
            matches!(strict, Err(SchemaError::UnsupportedInStrict(_))),
            "strict must 400 on {s}, got {strict:?}"
        );
        assert!(strict.as_ref().unwrap_err().is_unsupported_keyword());
        // Non-strict: must NOT error (degrades).
        assert!(
            SchemaNode::parse(s, false).is_ok(),
            "non-strict must accept (degrade) {s}"
        );
    }
}

#[test]
fn strict_accepts_supported_keywords() {
    // Supported keywords must NOT trip the strict guard.
    let ok: &[Value] = &[
        json!({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":false}),
        json!({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}),
        json!({"type":"string","enum":["a","b"]}),
        json!({"const":"x"}),
        json!({"oneOf":[{"const":"a"},{"const":"b"}]}),
    ];
    for s in ok {
        assert!(
            SchemaNode::parse(s, true).is_ok(),
            "strict must accept supported schema {s}"
        );
    }
}

// ── $defs / $ref local resolution ─────────────────────────────────

#[test]
fn ref_resolves_local_defs() {
    // `$ref` → `#/$defs/Color` resolves to the enum and enforces it.
    let schema = json!({
        "type":"object",
        "properties":{"color":{"$ref":"#/$defs/Color"}},
        "required":["color"],
        "$defs":{"Color":{"type":"string","enum":["red","green"]}}
    });
    let n = SchemaNode::parse(&schema, true).expect("ref resolves");
    let mut g = SchemaGrammar::new(n.clone());
    feed(&mut g, "{\"color\":\"red\"}").expect("conforming accepted");
    assert!(g.is_done());
    // Non-enum value rejected: after `"color":"`, only r/g allowed.
    let mut g2 = SchemaGrammar::new(n);
    feed(&mut g2, "{\"color\":\"").unwrap();
    assert!(
        g2.step(b'b').is_err(),
        "ref-resolved enum must reject `blue`"
    );
}

#[test]
fn ref_resolves_definitions_spelling() {
    // Draft-7 `definitions` spelling also resolves.
    let schema = json!({
        "type":"object",
        "properties":{"n":{"$ref":"#/definitions/Count"}},
        "definitions":{"Count":{"type":"integer"}}
    });
    let n = SchemaNode::parse(&schema, false).expect("definitions ref resolves");
    let mut g = SchemaGrammar::new(n);
    feed(&mut g, "{\"n\":42}").expect("integer accepted");
    assert!(g.is_done());
}

#[test]
fn ref_unresolvable_errors() {
    // Remote ref → 400 in both modes.
    assert!(matches!(
        SchemaNode::parse(&json!({"$ref":"https://x/y.json"}), false),
        Err(SchemaError::UnresolvableRef(_))
    ));
    // Dangling local ref → 400.
    assert!(matches!(
        SchemaNode::parse(
            &json!({"$ref":"#/$defs/Missing","$defs":{"Other":{"type":"string"}}}),
            false
        ),
        Err(SchemaError::UnresolvableRef(_))
    ));
}

#[test]
fn ref_recursion_is_depth_capped() {
    // Self-referential `$ref` must not blow the stack — depth cap
    // degrades (non-strict ⇒ Ok, strict ⇒ unsupported error).
    let schema = json!({
        "$ref":"#/$defs/Node",
        "$defs":{"Node":{"type":"object","properties":{"child":{"$ref":"#/$defs/Node"}}}}
    });
    // Non-strict: terminates with Ok (degrades at the cap).
    assert!(SchemaNode::parse(&schema, false).is_ok());
    // Strict: terminates with the depth-cap unsupported error.
    assert!(matches!(
        SchemaNode::parse(&schema, true),
        Err(SchemaError::UnsupportedInStrict(_))
    ));
}

#[test]
fn allof_single_branch_flattens() {
    // `allOf:[{$ref}]` — the idiomatic OpenAI wrapper — flattens to the
    // referenced node and enforces it.
    let schema = json!({
        "type":"object",
        "properties":{"status":{"allOf":[{"$ref":"#/$defs/Status"}]}},
        "$defs":{"Status":{"type":"string","enum":["on","off"]}}
    });
    let n = SchemaNode::parse(&schema, true).expect("single-branch allOf flattens");
    let mut g = SchemaGrammar::new(n);
    feed(&mut g, "{\"status\":\"on\"}").expect("enum via allOf+ref accepted");
    assert!(g.is_done());
}

#[test]
fn allof_multi_branch_degrades_or_errors() {
    let schema = json!({"allOf":[{"type":"object"},{"type":"object"}]});
    assert_eq!(node(&schema, false), SchemaNode::Any);
    assert!(matches!(
        SchemaNode::parse(&schema, true),
        Err(SchemaError::UnsupportedInStrict("allOf"))
    ));
}

#[test]
fn nested_object_schema() {
    let n = node(
        &json!({
            "type":"object",
            "properties":{
                "addr":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            },
            "required":["addr"]
        }),
        false,
    );
    let mut g = SchemaGrammar::new(n);
    feed(&mut g, "{\"addr\":{\"city\":\"Paris\"}}").expect("nested object accepted");
    assert!(g.is_done());
}

#[test]
fn property_conforming_values_accepted() {
    // Deterministic generator: 50 (schema, conforming value) pairs.
    // Seeded LCG, no external rng crate.
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed >> 33
    };
    let mut pass = 0;
    for _ in 0..50 {
        let pick = next() % 6;
        let (schema, value): (Value, Value) = match pick {
            0 => (
                json!({"type":"string"}),
                json!(format!("s{}", next() % 100)),
            ),
            1 => (json!({"type":"integer"}), json!((next() % 1000) as i64)),
            2 => (
                json!({"type":"string","enum":["red","green","blue"]}),
                json!(["red", "green", "blue"][(next() % 3) as usize]),
            ),
            3 => (
                json!({"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":4}),
                json!(vec![(next() % 9) as i64, (next() % 9) as i64]),
            ),
            4 => (
                json!({"type":"object","properties":{"n":{"type":"integer"},"s":{"type":"string"}},"required":["n","s"]}),
                json!({"n": (next()%50) as i64, "s": "ok"}),
            ),
            _ => (
                json!({"oneOf":[{"const":"yes"},{"const":"no"}]}),
                json!(["yes", "no"][(next() % 2) as usize]),
            ),
        };
        let n = SchemaNode::parse(&schema, false).expect("schema parses");
        let mut g = SchemaGrammar::new(n);
        let bytes = serde_json::to_string(&value).unwrap();
        let r = feed(&mut g, &bytes);
        assert!(
            r.is_ok() && g.is_done(),
            "conforming value {bytes} rejected by schema {schema}"
        );
        pass += 1;
    }
    assert_eq!(pass, 50);
}

// ── A6.5: scalar-root engagement tests ──────────────────────────────────────

/// Builds a minimal synthetic bytes map for SchemaConstraint tests.
fn synthetic_bm(entries: &[&[u8]]) -> Arc<TokenBytesMap> {
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

/// scalar-root string/enum: model emits bare `medium` first; constraint
/// must engage immediately on the first post-think token and force `"`.
/// After engagement, the grammar's mask will only allow `"` as the first
/// valid byte, so `medium` bytes are fed but the grammar rejects them and
/// clamps to Done — the important assertion is that engagement fired.
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn scalar_root_enum_immediate_engage() {
    // Vocab: 0="medium" (bare), 1=`"medium"` (quoted), 2=EOS (empty)
    let bm = synthetic_bm(&[b"medium", b"\"medium\"", b""]);
    let schema = json!({"type":"string","enum":["small","medium","large"]});
    let node = SchemaNode::parse(&schema, false).unwrap();
    assert!(node.is_scalar_root(), "enum string must be scalar");
    let mut c = SchemaConstraint::from_parts(bm.clone(), vec![2], node.clone(), None);
    // With Immediate policy, the FIRST step_mask call proactively engages
    // (no warm-up phase needed for scalar roots). The mask is NOT all-true
    // from the start — it immediately constrains to valid JSON starters
    // for the schema (here: only `"` is valid, so only id=1 is allowed).
    let m0_allowed = {
        let m = c.step_mask(3);
        (m[0], m[1], m[2])
    };
    assert!(c.engaged, "Immediate policy must engage at first step_mask");
    assert!(
        !m0_allowed.0,
        "bare `medium` must NOT be allowed — not a JSON starter"
    );
    assert!(
        m0_allowed.1,
        "`\"medium\"` must be allowed — starts with `\"`"
    );
    assert!(
        !m0_allowed.2,
        "EOS must not be allowed before any JSON value"
    );

    // Advance with token 1 (`"medium"`) — grammar accepts, value complete.
    c.advance(1);
    assert!(c.finished(), "`\"medium\"` completes the enum grammar");
    // Confirm the decoded value is valid JSON string.
    let v: Value = serde_json::from_str("\"medium\"").unwrap();
    assert_eq!(v.as_str(), Some("medium"));

    // Also verify: advance with token 0 (bare `medium`) first — token is
    // NOT a valid JSON starter so advance discards it but keeps engaged.
    let mut c2 = SchemaConstraint::from_parts(bm, vec![2], node, None);
    // Trigger engagement via step_mask first.
    let _ = c2.step_mask(3);
    assert!(c2.engaged, "engaged after first step_mask");
    // Advance with bare `medium` — engagement is already true, grammar at
    // Start, `m` is not a valid starter for the enum → grammar rejected,
    // Done flagged (this is the defensive path for post-engage unmasked tokens).
    // After Done, step_mask returns EOS-only.
    c2.advance(0); // should trigger grammar rejection, set done
    let m2 = c2.step_mask(3);
    // Grammar is Done or at some terminal state — only EOS should be allowed.
    assert!(m2[2], "EOS must be allowed when grammar is done/clamped");
}

/// Regression: leading whitespace before the root value must NOT be allowed.
/// JSON permits it, but treating WS as a no-op let a greedy (temp=0) decoder
/// loop on JSON-legal whitespace forever instead of emitting the value —
/// scalar/enum roots were the worst case. The mask must allow `"` but reject
/// whitespace-only tokens.
#[test]
fn scalar_root_rejects_leading_whitespace() {
    // Vocab: 0=`"` 1=space 2=newline 3=`  ` 4=`"small"` 5=EOS(empty)
    let bm = synthetic_bm(&[b"\"", b" ", b"\n", b"  ", b"\"small\"", b""]);
    let n = node(&json!({"type":"string","enum":["small","large"]}), false);
    let mut c = SchemaConstraint::from_parts(bm, vec![5], n, None);
    let (quote, space, newline, dbl_space, quoted_val) = {
        let m = c.step_mask(6);
        (m[0], m[1], m[2], m[3], m[4])
    };
    assert!(c.engaged, "Immediate policy engages at first step_mask");
    assert!(quote, "`\"` must be allowed at root value-start");
    assert!(
        quoted_val,
        "`\"small\"` must be allowed at root value-start"
    );
    assert!(!space, "single space must NOT be allowed (would loop)");
    assert!(!newline, "newline must NOT be allowed");
    assert!(!dbl_space, "double-space must NOT be allowed");
}

/// The root-whitespace rejection must NOT affect interior whitespace: a
/// pretty-printed object must still validate end-to-end.
#[test]
fn object_root_allows_interior_whitespace() {
    let n = node(
        &json!({"type":"object","properties":{"k":{"type":"string"}},"required":["k"],"additionalProperties":false}),
        true,
    );
    let mut g = SchemaGrammar::new(n);
    feed(&mut g, "{\n  \"k\": \"v\"\n}").expect("pretty-printed object must validate");
    assert!(g.is_done(), "object must complete");
}

/// Step a clone of `g` by one byte. A failed `step` leaves the real grammar's
/// `leaf` taken, so probing several candidate bytes from one state needs a
/// fresh clone each time.
fn probe_byte(g: &SchemaGrammar, b: u8) -> Result<(), ()> {
    let mut c = g.clone();
    c.step(b)
}

/// Regression: whitespace *inside* an enum/literal must be rejected, not
/// skipped. After the opening `"` the grammar is mid-literal; treating a
/// newline there as a no-op let a greedy decoder loop on `"\n\n…` forever.
#[test]
fn enum_literal_rejects_interior_whitespace() {
    let n = node(&json!({"type":"string","enum":["small","large"]}), false);
    let mut base = SchemaGrammar::new(n);
    base.step(b'"')
        .expect("opening quote starts the enum literal");
    assert!(
        probe_byte(&base, b'\n').is_err(),
        "newline mid-literal rejected"
    );
    assert!(
        probe_byte(&base, b' ').is_err(),
        "space mid-literal rejected"
    );
    assert!(
        probe_byte(&base, b's').is_ok(),
        "`s` advances toward \"small\""
    );
}

/// Regression: a raw (unescaped) control char inside a string is illegal JSON
/// and must be rejected — this also stops the whitespace loop for free-form
/// `type:string` values. Printable bytes (incl. space) stay valid content.
#[test]
fn free_string_rejects_raw_control_chars() {
    let n = node(&json!({"type":"string"}), false);
    let mut base = SchemaGrammar::new(n);
    base.step(b'"').expect("opening quote starts the string");
    assert!(
        probe_byte(&base, b'\n').is_err(),
        "raw newline rejected in string"
    );
    assert!(
        probe_byte(&base, b'\t').is_err(),
        "raw tab rejected in string"
    );
    assert!(
        probe_byte(&base, b'a').is_ok(),
        "ordinary char is valid content"
    );
    assert!(
        probe_byte(&base, b' ').is_ok(),
        "space (0x20) is valid content"
    );
}

/// scalar-root integer: the constraint must engage immediately and the
/// grammar must reject `.` (integer forbids decimal point).
#[test]
fn scalar_root_integer_immediate_engage() {
    // Vocab: 0="4" (digit), 1="." (dot), 2=EOS (empty)
    let bm = synthetic_bm(&[b"4", b".", b""]);
    let schema = json!({"type":"integer"});
    let node = SchemaNode::parse(&schema, false).unwrap();
    assert!(node.is_scalar_root(), "integer must be scalar");
    let mut c = SchemaConstraint::from_parts(bm, vec![2], node, None);
    // Engage on first token (digit `4`).
    c.advance(0);
    assert!(
        c.engaged,
        "Immediate policy must engage on first digit token"
    );
    // After `4`, `.` must be blocked (integer).
    let m = c.step_mask(3);
    assert!(!m[1], "`.` must be disallowed after integer digit");
    assert!(
        m[2],
        "EOS must be allowed — integer `4` is complete at top level"
    );
}

/// scalar-root boolean: the constraint must engage immediately and the
/// grammar must allow `true`/`false` only.
#[test]
fn scalar_root_boolean_immediate_engage() {
    // Vocab: 0="true", 1="false", 2="maybe"(invalid), 3=EOS (empty)
    let bm = synthetic_bm(&[b"true", b"false", b"maybe", b""]);
    let schema = json!({"type":"boolean"});
    let node = SchemaNode::parse(&schema, false).unwrap();
    assert!(node.is_scalar_root(), "boolean must be scalar");
    let mut c = SchemaConstraint::from_parts(bm, vec![3], node, None);
    // Immediate policy: engage on first token.
    c.advance(0); // token 0 = "true"
    assert!(c.engaged, "Immediate policy engaged");
    assert!(c.finished(), "`true` completes boolean grammar");
}

/// Schema fence suppression: pre-engagement buffer detects fence.
#[test]
fn schema_fence_suppression_detected() {
    let bm = synthetic_bm(&[b"```json\n", b"\"ok\"", b""]);
    let schema = json!({"type":"string","enum":["ok","bad"]});
    let node = SchemaNode::parse(&schema, false).unwrap();
    let mut c = SchemaConstraint::from_parts(bm, vec![2], node, None);
    // Token 0 is fence — Immediate policy engages immediately, but we want
    // to check that the fence IS detected in pre_engage_buf before engagement.
    // Reset engaged to test pre-buf accumulation via a non-Immediate path.
    // We'll directly inspect the pre_engage_buf via pre_engage_is_fence.
    // For the Immediate policy, pre_engage_buf is populated then engagement
    // fires on the same token. The fence must be detectable.
    c.advance(0); // token 0 = "```json\n" — not valid JSON, grammar rejects and clamps
                  // pre_engage_buf should hold the fence text.
    assert!(
        c.pre_engage_is_fence(),
        "fence prefix must be detected in pre_engage_buf"
    );
}

/// fence_suppression negative: real prose before JSON is NOT flagged as fence.
#[test]
fn schema_fence_suppression_prose_not_fence() {
    let bm = synthetic_bm(&[b"Sure, here: ", b"{", b"}", b""]);
    let schema = json!({"type":"object","properties":{"x":{"type":"integer"}}});
    let node = SchemaNode::parse(&schema, false).unwrap();
    // ValueStarter policy (object root).
    assert!(!node.is_scalar_root());
    let mut c = SchemaConstraint::from_parts(bm, vec![3], node, None);
    c.advance(0); // prose token — ValueStarter, no engagement yet
    assert!(!c.engaged, "no engagement on prose before `{{`");
    assert!(
        !c.pre_engage_is_fence(),
        "real prose must NOT be flagged as fence"
    );
}

/// object/array roots keep ValueStarter policy (regression guard).
#[test]
fn object_array_root_keep_value_starter_policy() {
    let obj_node = SchemaNode::parse(&json!({"type":"object"}), false).unwrap();
    let arr_node =
        SchemaNode::parse(&json!({"type":"array","items":{"type":"integer"}}), false).unwrap();
    assert!(!obj_node.is_scalar_root(), "object is not scalar");
    assert!(!arr_node.is_scalar_root(), "array is not scalar");

    let bm = synthetic_bm(&[b"AAAA", b"{", b"[", b"]", b"}", b""]);
    let mut c_obj = SchemaConstraint::from_parts(bm, vec![5], obj_node, None);
    // ValueStarter: non-structural token must NOT engage.
    c_obj.advance(0);
    assert!(
        !c_obj.engaged,
        "object root must not engage on non-structural token"
    );
    // `{` must engage.
    c_obj.advance(1);
    assert!(c_obj.engaged, "object root must engage on `{{`");
}

/// is_scalar_root classification covers all expected types.
#[test]
fn is_scalar_root_classification() {
    assert!(SchemaNode::Str { enum_: None }.is_scalar_root());
    assert!(SchemaNode::Str {
        enum_: Some(vec!["a".into()])
    }
    .is_scalar_root());
    assert!(SchemaNode::Num { integer: false }.is_scalar_root());
    assert!(SchemaNode::Num { integer: true }.is_scalar_root());
    assert!(SchemaNode::Bool.is_scalar_root());
    assert!(SchemaNode::Null.is_scalar_root());
    assert!(SchemaNode::Const(json!("hello")).is_scalar_root());
    // Union of scalars → scalar
    assert!(SchemaNode::Union(vec![
        SchemaNode::Const(json!("a")),
        SchemaNode::Const(json!("b")),
    ])
    .is_scalar_root());
    // Union with container → not scalar
    assert!(!SchemaNode::Union(vec![
        SchemaNode::Const(json!("a")),
        SchemaNode::Object {
            props: Arc::from([]),
            required: Arc::from([]),
            additional: false
        },
    ])
    .is_scalar_root());
    assert!(!SchemaNode::Object {
        props: Arc::from([]),
        required: Arc::from([]),
        additional: false
    }
    .is_scalar_root());
    assert!(!SchemaNode::Array {
        items: Arc::new(SchemaNode::Num { integer: false }),
        min: None,
        max: None
    }
    .is_scalar_root());
    assert!(!SchemaNode::Any.is_scalar_root());
}

// ── engage_policy_override + thinking-suppress tests ────────────────────────

/// Object-root schema with `force_engage_policy=Some(Immediate)` must
/// engage on the FIRST token even though the token is not `{`.
///
/// This mirrors the tool_choice=required/named path where the model (e.g.
/// Gemma4) may emit a prefix before the `{`-byte of the JSON object.
#[test]
fn object_root_immediate_override_engages_at_token_1() {
    // Vocab: 0="<|tool_call|>call:fn" (prefix, no `{`), 1="{", 2="}", 3=EOS
    let bm = synthetic_bm(&[b"<|tool_call|>call:fn", b"{", b"}", b""]);
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    });
    let node = SchemaNode::parse(&schema, false).unwrap();
    // Sanity: without override, object root uses ValueStarter.
    assert!(!node.is_scalar_root(), "object root is not scalar");

    let mut c = SchemaConstraint::from_parts(
        bm,
        vec![3],
        node,
        Some(EngagePolicy::Immediate), // force Immediate for tool_choice
    );

    // First step_mask with Immediate policy must proactively engage (flag
    // is_thinking == false initially → no deferral).
    let _ = c.step_mask(4);
    assert!(
        c.engaged,
        "Immediate override: object-root constraint must engage at first step_mask"
    );
}

/// When `is_thinking` handle is NOT wired (None passed to
/// GenerationRequest), the constraint's `is_thinking` stays false and
/// Immediate policy fires on token 1 even for a thinking model whose
/// ThinkSplitter starts open.
///
/// This test directly asserts that starting with `is_thinking == false`
/// (the initial value of the AtomicBool) and NOT storing into it means
/// Immediate engage fires on the first step_mask call.
#[test]
fn immediate_engage_when_thinking_handle_absent() {
    // Vocab: 0="reasoning text", 1="{", 2="}", 3=EOS
    let bm = synthetic_bm(&[b"reasoning text", b"{", b"}", b""]);
    let schema = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "arguments": {"type": "object"}
        },
        "required": ["name"],
        "additionalProperties": false
    });
    let node = SchemaNode::parse(&schema, false).unwrap();
    // Immediate override — handle NOT connected to any engine (simulates bare_json_tool_call_mode).
    let mut c = SchemaConstraint::from_parts(bm, vec![3], node, Some(EngagePolicy::Immediate));
    // is_thinking starts false (AtomicBool::new(false) in constructor).
    // No engine writes to the handle. First step_mask must engage.
    let _ = c.step_mask(4);
    assert!(
        c.engaged,
        "Immediate with no thinking handle: must engage on first step_mask"
    );
}

/// The `engage_policy_override=None` path (response_format path)
/// keeps the derived policy — object root → ValueStarter, not Immediate.
/// This is the regression guard ensuring the response_format path is unchanged.
#[test]
fn object_root_none_override_keeps_value_starter() {
    // Vocab: 0="Sure!", 1="{", 2="}", 3=EOS
    let bm = synthetic_bm(&[b"Sure!", b"{", b"}", b""]);
    let schema = json!({
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": false
    });
    let node = SchemaNode::parse(&schema, false).unwrap();

    let mut c = SchemaConstraint::from_parts(bm, vec![3], node, None);
    // ValueStarter: prose token "Sure!" must NOT trigger engagement.
    let m0 = c.step_mask(4);
    assert!(
        m0.iter().all(|&b| b),
        "ValueStarter warmup: all tokens allowed before `{{`-byte"
    );
    assert!(
        !c.engaged,
        "ValueStarter: must NOT engage on non-structural token"
    );
}

/// `reset_from` (the buffer-reusing per-token reset) must restore a scratch
/// grammar to a state byte-identical to a fresh `clone()` of the reference,
/// regardless of how the scratch was dirtied first. This guards the buffer
/// reuse in `reset_from` / `Frame::reset_from` / `LiteralTrie::reset_from`
/// against leaving stale mutable progress behind.
#[test]
fn reset_from_matches_fresh_clone() {
    use super::super::ProbeGrammar;

    // A schema rich enough to exercise object frames (with `emitted`), an
    // enum literal trie, and array frames.
    let root = node(
        &json!({
            "type":"object",
            "properties":{
                "name":{"type":"string"},
                "size":{"type":"string","enum":["small","medium","large"]},
                "tags":{"type":"array","items":{"type":"string"}}
            },
            "required":["name","size","tags"],
            "additionalProperties":false
        }),
        true,
    );

    // Reference states at several distinct grammar positions.
    let prefixes = [
        "{",
        "{\"name\":\"",
        "{\"name\":\"x\",\"size\":\"",
        "{\"name\":\"x\",\"size\":\"small\",\"tags\":[\"",
    ];
    // Bytes that drive the scratch into a different shape before each reset.
    let dirtiers = [b'x', b'{', b'"', b'}', b'[', b' ', b',', b':', b']'];

    for prefix in prefixes {
        let mut base = SchemaGrammar::new(root.clone());
        feed(&mut base, prefix).expect("prefix valid");
        let expect = format!("{base:?}");
        let expect_allowed = base.allowed_bytes();

        let mut scratch = base.clone();
        for &d in &dirtiers {
            let _ = scratch.step(d); // diverge (may Err / no-op; that's fine)
            scratch.reset_from(&base);
            assert_eq!(
                format!("{scratch:?}"),
                expect,
                "reset_from must restore base after stepping {:?} at prefix {prefix:?}",
                d as char
            );
            assert_eq!(
                scratch.allowed_bytes(),
                expect_allowed,
                "reset_from allowed_bytes mismatch at prefix {prefix:?}"
            );
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Perf probe: SchemaGrammar step_mask cost, across three columns.
//
// The immutable schema inside each frame is `Arc`-shared, so a grammar clone
// no longer deep-copies the property list / element schema / literal set —
// only the tiny mutable progress. This measures, on a synthetic ~152K vocab:
//   total_ms      — full per-token sweep (reset + step), the user-visible cost
//   cloneOnly_ms  — a raw `clone()` per token (no stepping)
//   reset_ms      — the production path: one buffer-reusing `reset_from` per
//                   token (what `fill_allow_mask` actually does)
// Compare to the json_object probe (~1-3 ms/step).
//
// Run: cargo test -p rmlx-server constraint_json::schema::tests::probe_schema \
//        --profile release-perf -- --ignored --nocapture
// ───────────────────────────────────────────────────────────────────────────

struct LcgS(u64);
impl LcgS {
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

#[allow(
    clippy::indexing_slicing,
    reason = "letter index bounded by rng.below(letters.len())"
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
        b"-",
        b"true",
        b"false",
        b"null",
        b"\"name\"",
        b"\"color\"",
        b"\"size\"",
        b"small",
        b"medium",
    ];
    let mut owned: Vec<Vec<u8>> = structural.iter().map(|s| s.to_vec()).collect();
    let mut rng = LcgS(0x1234_5678_9abc_def0);
    let letters = b"abcdefghijklmnopqrstuvwxyz";
    while owned.len() < n {
        let len = match rng.below(100) {
            0..=14 => 1,
            15..=39 => 2,
            40..=64 => 4,
            65..=84 => 6,
            _ => 9,
        };
        let mut t: Vec<u8> = Vec::with_capacity(len + 1);
        match rng.below(100) {
            0..=19 => t.push(b' '),
            20..=24 => t.push(b'"'),
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
    synthetic_bm(&refs)
}

#[test]
#[ignore = "perf probe — run manually with --ignored --nocapture"]
fn probe_schema_step_mask_cost() {
    let vocab_n = 152_064usize;
    let bm = realistic_vocab(vocab_n);
    let n = bm.vocab_size();
    let reps = 30u32;

    // A realistic strict object: 6 properties incl. an enum. Frames on the
    // stack carry these property schemas → the deep-clone tax.
    let root = node(
        &json!({
            "type":"object",
            "properties":{
                "name":{"type":"string"},"color":{"type":"string"},
                "size":{"type":"string","enum":["small","medium","large"]},
                "count":{"type":"integer"},"ripe":{"type":"boolean"},
                "origin":{"type":"string"}
            },
            "required":["name","color","size","count","ripe","origin"],
            "additionalProperties":false
        }),
        true,
    );

    // (label, valid JSON prefix → reaches the probed grammar state)
    let scenes: &[(&str, &str)] = &[
        ("obj ExpectKey (6 props)", "{"),
        ("in string value", "{\"name\":\""),
        (
            "enum value-start (InLit)",
            "{\"name\":\"a\",\"color\":\"b\",\"size\":\"",
        ),
    ];

    println!("\nvocab={n}  reps={reps}  (per decode step = total/reps)\n");
    println!(
        "{:<32} {:>10} {:>12} {:>9} {:>11}",
        "grammar state", "total_ms", "cloneOnly_ms", "clone%", "reset_ms"
    );
    println!("{}", "-".repeat(78));

    for (label, prefix) in scenes {
        let mut base = SchemaGrammar::new(root.clone());
        feed(&mut base, prefix).expect("prefix must be valid for the schema");
        let mut mask = vec![false; n];

        // A) clone-per-token + step (current fill_allow_mask behaviour).
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            mask.fill(false);
            for (id, m) in mask.iter_mut().enumerate() {
                let bytes = bm.token_bytes(id);
                if bytes.is_empty() {
                    continue;
                }
                let mut g = base.clone();
                let mut ok = true;
                for &b in bytes {
                    if g.step(b).is_err() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    *m = true;
                }
            }
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

        // B) clone-only — the deep-copy tax (no stepping).
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            for id in 0..n {
                if bm.token_bytes(id).is_empty() {
                    continue;
                }
                let g = base.clone();
                std::hint::black_box(&g);
            }
        }
        let clone_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

        // C) reset_from — the production hot path. `fill_allow_mask` resets a
        // single scratch grammar per candidate token instead of cloning, so
        // this is the figure that actually governs constrained-decode cost.
        // Step one byte before each reset so the scratch shape diverges and the
        // reset must restore it (worst case for buffer reuse).
        let mut scratch = base.clone();
        let t2 = std::time::Instant::now();
        for _ in 0..reps {
            for id in 0..n {
                if bm.token_bytes(id).is_empty() {
                    continue;
                }
                let _ = scratch.step(b'x');
                <SchemaGrammar as super::super::ProbeGrammar>::reset_from(&mut scratch, &base);
                std::hint::black_box(&scratch);
            }
        }
        let reset_ms = t2.elapsed().as_secs_f64() * 1e3 / f64::from(reps);

        let pct = clone_ms / total_ms.max(1e-9) * 100.0;
        println!("{label:<32} {total_ms:>10.3} {clone_ms:>12.3} {pct:>8.1}% {reset_ms:>11.3}");
    }
    println!();
}
