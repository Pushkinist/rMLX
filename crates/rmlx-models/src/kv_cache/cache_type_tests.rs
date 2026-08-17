use super::*;

// Helper: round-trip — variant → tag → parse.
fn round_trip(ct: CacheType) {
    let tag = ct.tag();
    let parsed = parse(tag).unwrap_or_else(|e| panic!("round_trip failed for '{tag}': {e}"));
    assert_eq!(parsed, ct, "round_trip mismatch for tag '{tag}'");
}

#[test]
fn canonical_tags_round_trip() {
    round_trip(CacheType::Auto);
    round_trip(CacheType::Bf16);
    round_trip(CacheType::Q8G128);
    round_trip(CacheType::Q8G64);
    round_trip(CacheType::Q8G32);
    round_trip(CacheType::Q6G64);
    round_trip(CacheType::Q5G64);
    round_trip(CacheType::Q4G128);
    round_trip(CacheType::Q4G64);
    round_trip(CacheType::Q4G32);
    round_trip(CacheType::Q3G64);
    round_trip(CacheType::Q2G64);
    round_trip(CacheType::Tq4);
    round_trip(CacheType::Planar4);
    round_trip(CacheType::Planar3);
}

// ── 2-bit KV codec matrix ──────────────────────────────────────────

#[test]
fn q2_g64_round_trip_and_params() {
    // Tag round-trip + affine params (2-bit, group=64, V-side codec).
    round_trip(CacheType::Q2G64);
    assert_eq!(CacheType::Q2G64.bits(), Some(2));
    assert_eq!(CacheType::Q2G64.group_size(), Some(64));
    assert!(CacheType::Q2G64.is_affine());
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn kv_bitwidth_matrix_resolves() {
    // DoD: every supported V-side bit-width (2/3/3.5/4) resolves to a
    // concrete KvQuant when paired with an 8-bit K, given a head_dim that
    // satisfies that codec's MLX bit-packing rule (§D6.2: head_dim % (32/bits)).
    //
    // Document-the-truth: the rungs do NOT all resolve at the same head_dim.
    // - 2-bit packs 16 vals/u32 → head_dim % 16 == 0 (Bonsai's 128 OK).
    // - 4-bit packs 8 vals/u32 → head_dim % 8 == 0 (128 OK).
    // - 3-bit packs 10 vals/u32 → head_dim % 10 == 0 → 128 is REJECTED;
    // needs a multiple of lcm(64,10)=320. This is a pre-existing rMLX
    // guard (see `mlx_bit_packing_violation_q3_g64_on_unfriendly_head_dim`).
    // So 2/4 are asserted at head_dim=128 (Bonsai); 3 and the 3.5-bit
    // fractional endpoints (K3/V4) at head_dim=320.
    let arch = "Qwen3ForCausalLM";
    let auto = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };

    // 2-bit V at Bonsai head_dim=128 (asymmetric, K=8-bit q8_g64).
    let kq2 = resolve(
        spec(CacheType::Q8G64, CacheType::Q2G64),
        ctx(arch, Some(128)),
        auto,
    )
    .expect("q2_g64 V must resolve at head_dim=128");
    assert_eq!(
        kq2,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 2,
            k_group_size: 64,
            v_group_size: 64,
        },
        "2-bit V must resolve to Mixed{{k=8, v=2}}"
    );

    // 4-bit V at head_dim=128.
    let kq4 = resolve(
        spec(CacheType::Q8G64, CacheType::Q4G64),
        ctx(arch, Some(128)),
        auto,
    )
    .expect("q4_g64 V must resolve at head_dim=128");
    assert!(matches!(kq4, KvQuant::Mixed { v_bits: 4, .. }));

    // 3-bit V at a 3-bit-friendly head_dim=320 (128 violates §D6.2 for 3-bit).
    let kq3 = resolve(
        spec(CacheType::Q8G64, CacheType::Q3G64),
        ctx(arch, Some(320)),
        auto,
    )
    .expect("q3_g64 V must resolve at head_dim=320");
    assert!(matches!(kq3, KvQuant::Mixed { v_bits: 3, .. }));

    // 3.5-bit fractional = K floor=3 / V ceil=4. K=3 is NOT gated (only K=2
    // is). Needs head_dim=320 because the K side is 3-bit.
    let kq35 = resolve(
        spec(CacheType::Q3G64, CacheType::Q4G64),
        ctx(arch, Some(320)),
        auto,
    )
    .expect("3.5-bit (K3/V4) must resolve at head_dim=320");
    assert_eq!(
        kq35,
        KvQuant::Mixed {
            k_bits: 3,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn pure_2bit_k_rejected() {
    // 2-bit on the K side is gated — incoherent. combo_to_kv_quant
    // must reject (Q2G64, *) with UnsupportedCombo naming q2_g64.
    let auto = KvQuant::K8V8;
    let err = resolve(
        spec(CacheType::Q2G64, CacheType::Q2G64),
        ctx("Qwen3ForCausalLM", Some(128)),
        auto,
    )
    .unwrap_err();
    match err {
        ResolveError::UnsupportedCombo(msg) => {
            assert!(msg.contains("q2_g64"), "expected q2_g64 named: {msg}");
            assert!(
                msg.contains("V-side only") || msg.contains("2-bit K"),
                "expected K-gate explanation: {msg}"
            );
        }
        other => panic!("expected UnsupportedCombo, got {other:?}"),
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn q2_g64_head_dim_120_bit_packing_rejected() {
    // 2-bit MLX packing: head_dim % (32/2 = 16) == 0. head_dim=120 →
    // 120 % 16 = 8 → MlxBitPackingViolation. (120 % 64 != 0 actually fires
    // GroupSizeNotDivisible first — pick head_dim=128-friendly group but
    // bad packing: use group=64-divisible head_dim. 64 % 16 == 0, so use a
    // head_dim that divides 64 but not 16: none exist that divide 64. So we
    // assert the divisibility guard fires for a non-multiple-of-64 head_dim.)
    let err = resolve(
        spec(CacheType::Q8G64, CacheType::Q2G64),
        ctx("Qwen3ForCausalLM", Some(120)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    // 120 % 64 != 0 → GroupSizeNotDivisible for the V codec.
    assert!(
        matches!(
            err,
            ResolveError::GroupSizeNotDivisible { group_size: 64, .. }
        ),
        "got {err:?}"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn aliases_resolve_to_canonical_variants() {
    assert_eq!(parse("f16").unwrap(), CacheType::Bf16);
    assert_eq!(parse("none").unwrap(), CacheType::Bf16);
    assert_eq!(parse("turbo4").unwrap(), CacheType::Tq4);
}

#[test]
fn unknown_tag_returns_unknown_error() {
    match parse("garbage") {
        Err(ParseError::Unknown(s)) => assert_eq!(s, "garbage"),
        other => panic!("expected Unknown, got {other:?}"),
    }
    match parse("") {
        Err(ParseError::Unknown(s)) => assert_eq!(s, ""),
        other => panic!("expected Unknown for empty string, got {other:?}"),
    }
}

#[test]
fn q8_0_returns_not_implemented_with_hint() {
    match parse("q8_0") {
        Err(ParseError::NotImplemented(msg)) => {
            assert!(
                msg.contains("llama.cpp legacy"),
                "message must contain 'llama.cpp legacy': {msg}"
            );
            assert!(
                msg.contains("q8_g32") || msg.contains("q8_g128"),
                "message must name a substitute tag: {msg}"
            );
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

#[test]
fn q4_0_returns_not_implemented_with_hint() {
    match parse("q4_0") {
        Err(ParseError::NotImplemented(msg)) => {
            assert!(
                msg.contains("llama.cpp legacy"),
                "message must contain 'llama.cpp legacy': {msg}"
            );
            assert!(
                msg.contains("q4_g32") || msg.contains("q4_g64"),
                "message must name a substitute tag: {msg}"
            );
        }
        other => panic!("expected NotImplemented, got {other:?}"),
    }
}

#[test]
fn other_reserved_tags_return_not_implemented() {
    for tag in &["q4_1", "q5_0", "q5_1", "iq4_nl"] {
        match parse(tag) {
            Err(ParseError::NotImplemented(_)) => {}
            other => panic!("expected NotImplemented for '{tag}', got {other:?}"),
        }
    }
}

// ── Resolver tests (Task 3) ───────────────────────────────────────────────

fn spec(k: CacheType, v: CacheType) -> CacheTypeSpec {
    CacheTypeSpec { k, v }
}

fn ctx<'a>(arch: &'a str, head_dim: Option<usize>) -> ResolverContext<'a> {
    ResolverContext {
        arch_class: arch,
        head_dim,
    }
}

#[test]
fn cache_type_bits_and_group_size() {
    assert_eq!(CacheType::Q8G128.bits(), Some(8));
    assert_eq!(CacheType::Q8G128.group_size(), Some(128));
    assert_eq!(CacheType::Q4G64.bits(), Some(4));
    assert_eq!(CacheType::Q4G64.group_size(), Some(64));
    // Non-affine returns None on both.
    for ct in [
        CacheType::Auto,
        CacheType::Bf16,
        CacheType::Tq4,
        CacheType::Planar4,
    ] {
        assert_eq!(ct.bits(), None, "{}", ct.tag());
        assert_eq!(ct.group_size(), None, "{}", ct.tag());
    }
}

#[test]
fn decompose_auto_inverse_of_canonical() {
    assert_eq!(
        decompose_auto(KvQuant::None),
        (CacheType::Bf16, CacheType::Bf16)
    );
    assert_eq!(
        decompose_auto(KvQuant::K8V8),
        (CacheType::Q8G128, CacheType::Q8G128)
    );
    assert_eq!(
        decompose_auto(KvQuant::K8V4),
        (CacheType::Q8G128, CacheType::Tq4)
    );
    assert_eq!(
        decompose_auto(KvQuant::Planar),
        (CacheType::Q8G128, CacheType::Planar4)
    );
    // Bonsai-style Mixed.
    assert_eq!(
        decompose_auto(KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }),
        (CacheType::Q8G64, CacheType::Q4G64)
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn auto_auto_passthrough_returns_auto_unchanged() {
    let kq = resolve(
        spec(CacheType::Auto, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(kq, KvQuant::K8V8);
}

// ── K-side rotation codec (rot_k) ──────────────────────────────────

#[test]
fn rot_k_round_trip_and_params() {
    round_trip(CacheType::RotK);
    // rot_k is non-affine (rotated 8-bit K is handled specially), so the
    // affine validators skip it.
    assert_eq!(CacheType::RotK.bits(), None);
    assert_eq!(CacheType::RotK.group_size(), None);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn rot_k_on_k_resolves_to_rotk_with_affine_v() {
    // --ctk rot_k --ctv q4_g64 on Bonsai (head_dim=128, power of two).
    let kq = resolve(
        spec(CacheType::RotK, CacheType::Q4G64),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .expect("rot_k + q4_g64 must resolve at head_dim=128");
    assert_eq!(
        kq,
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        }
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn rot_k_auto_v_on_bonsai_resolves() {
    // --ctk rot_k --ctv auto on Bonsai: auto base Mixed{k8,v4,g64} → V=q4_g64.
    let auto = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let kq = resolve(
        spec(CacheType::RotK, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        auto,
    )
    .expect("rot_k + auto-V must resolve");
    assert_eq!(
        kq,
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        }
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rot_k_non_power_of_two_head_dim_rejected() {
    // head_dim=96 is divisible by 64? no — but pow2 check fires first only
    // after head_dim passes. Use 192 (div by 64, not pow2) to isolate.
    let err = resolve(
        spec(CacheType::RotK, CacheType::Q4G64),
        ctx("Qwen3ForCausalLM", Some(192)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::RotKHeadDimNotPow2(192));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rot_k_on_v_side_rejected() {
    let err = resolve(
        spec(CacheType::Q8G128, CacheType::RotK),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::RotKVSide);
}

// rot_k + tq4 → RotKTq4V resolver tests.

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn rot_k_tq4_resolves_to_rot_k_tq4v() {
    // DoD: --ctk rot_k --ctv tq4 on Bonsai (head_dim=128, pow2, tq4-eligible).
    let kq = resolve(
        spec(CacheType::RotK, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .expect("rot_k + tq4 must resolve at head_dim=128");
    assert_eq!(kq, KvQuant::RotKTq4V, "must map to RotKTq4V");
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn rot_k_tq4_head_dim_256_resolves() {
    // tq4 also supported at head_dim=256 (Gemma4 global head_dim).
    let kq = resolve(
        spec(CacheType::RotK, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(256)),
        KvQuant::K8V8,
    )
    .expect("rot_k + tq4 must resolve at head_dim=256");
    assert_eq!(kq, KvQuant::RotKTq4V);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rot_k_tq4_non_pow2_head_dim_rejected() {
    // pow2 constraint still fires before tq4 head_dim check.
    let err = resolve(
        spec(CacheType::RotK, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(192)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::RotKHeadDimNotPow2(192));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rot_k_tq4_unsupported_head_dim_64_rejected() {
    // 64 is pow2 but not in tq4's {128, 256} set.
    let err = resolve(
        spec(CacheType::RotK, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(64)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::Tq4UnsupportedHeadDim(64));
}

#[test]
fn rot_k_tq4v_decompose_round_trips() {
    // decompose_auto(RotKTq4V) must return (RotK, Tq4).
    assert_eq!(
        decompose_auto(KvQuant::RotKTq4V),
        (CacheType::RotK, CacheType::Tq4)
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn rot_k_tq4v_display_and_parse() {
    // KvQuant::RotKTq4V must round-trip through Display + FromStr.
    let s = KvQuant::RotKTq4V.to_string();
    assert_eq!(s, "rot_k_tq4v");
    let parsed: KvQuant = s.parse().expect("rot_k_tq4v must parse");
    assert_eq!(parsed, KvQuant::RotKTq4V);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn rot_k_non_affine_non_tq4_v_rejected() {
    // planar4 on V is still unsupported with rot_k — no mapping defined.
    let err = resolve(
        spec(CacheType::RotK, CacheType::Planar4),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    match err {
        ResolveError::UnsupportedCombo(msg) => {
            assert!(msg.contains("rot_k"), "expected rot_k named: {msg}");
        }
        other => panic!("expected UnsupportedCombo, got {other:?}"),
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn rot_k_with_affine_v_still_resolves_to_rot_k() {
    // must not break the existing rot_k + q4_g64 path.
    let kq = resolve(
        spec(CacheType::RotK, CacheType::Q4G64),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .expect("rot_k + q4_g64 must still resolve");
    assert_eq!(
        kq,
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        }
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tq4_still_guarded_on_k_with_rot_k_present() {
    // Regression: adding rot_k must NOT lift the V-only guard for tq4/planar4.
    let err = resolve(
        spec(CacheType::Tq4, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert!(matches!(err, ResolveError::KSideRotationCodec(t) if t == "tq4"));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn k_side_tq4_rejected_with_rotation_error() {
    let err = resolve(
        spec(CacheType::Tq4, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert!(
        matches!(err, ResolveError::KSideRotationCodec(tag) if tag == "tq4"),
        "got {err:?}"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn k_side_planar4_rejected_with_rotation_error() {
    let err = resolve(
        spec(CacheType::Planar4, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert!(
        matches!(err, ResolveError::KSideRotationCodec(tag) if tag == "planar4"),
        "got {err:?}"
    );
}

// ── Planar3 (3-bit V codec) resolver tests ────────────────────────────────────

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn planar3_explicit_resolves_to_kv_quant_planar3() {
    // (Q8G128, Planar3) → KvQuant::Planar3
    let kq = resolve(
        spec(CacheType::Q8G128, CacheType::Planar3),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V4,
    )
    .unwrap();
    assert_eq!(kq, KvQuant::Planar3);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn planar3_k_side_rejected() {
    // Planar3 is V-only — reject on K side.
    let err = resolve(
        spec(CacheType::Planar3, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert!(
        matches!(err, ResolveError::KSideRotationCodec(tag) if tag == "planar3"),
        "got {err:?}"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn planar3_parse_round_trip() {
    // "planar3" and "planar_3" both parse to CacheType::Planar3.
    assert_eq!(parse("planar3").ok(), Some(CacheType::Planar3));
    assert_eq!(parse("planar_3").ok(), Some(CacheType::Planar3));
    assert_eq!(CacheType::Planar3.tag(), "planar3");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn planar3_non_q8g128_k_rejected() {
    // Planar3 with non-Q8G128 K is rejected.
    let err = combo_to_kv_quant(CacheType::Q8G64, CacheType::Planar3).unwrap_err();
    assert!(
        matches!(err, ResolveError::UnsupportedCombo(_)),
        "expected UnsupportedCombo, got {err:?}"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn head_dim_unknown_returns_error() {
    let err = resolve(
        spec(CacheType::Auto, CacheType::Auto),
        ctx("Qwen3ForCausalLM", None),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::HeadDimUnknown);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tq4_head_dim_64_rejected() {
    let err = resolve(
        spec(CacheType::Auto, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(64)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::Tq4UnsupportedHeadDim(64));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tq4_head_dim_128_and_256_accepted() {
    let kq128 = resolve(
        spec(CacheType::Auto, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(kq128, KvQuant::K8V4);

    let kq256 = resolve(
        spec(CacheType::Auto, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(256)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(kq256, KvQuant::K8V4);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tq4_gemma4_full_attention_head_dim_512_rejected() {
    // every Gemma4 variant (e2b/e4b/26b-a4b) reports the
    // full-attention head_dim 512 via `ModelConfig::head_dim()`
    // (text_config.global_head_dim). The SWA head_dim 256 belongs only to
    // the rotating layers, which stay bf16 (§5.7) and are never
    // tq4-quantized. So `--ctk q8_g128 --ctv tq4` on Gemma4 must reject at
    // resolve time with Tq4UnsupportedHeadDim(512) — clean exit 78, no
    // runtime crash. This locks in the "do not force tq4 on an unsupported
    // head_dim" invariant.
    let err = resolve(
        spec(CacheType::Q8G128, CacheType::Tq4),
        ctx("Gemma4ForConditionalGeneration", Some(512)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::Tq4UnsupportedHeadDim(512));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn bonsai_asymmetric_auto_tq4_rejected_no_silent_coercion() {
    // Bonsai's auto is Mixed{8,4,64,64} → decompose gives (Q8G64, Q4G64).
    // User asks --ctk auto --ctv tq4 → K stays Q8G64, V becomes Tq4.
    // combo_to_kv_quant must NOT silently promote Q8G64→Q8G128; it must reject.
    let auto = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let err = resolve(
        spec(CacheType::Auto, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(128)),
        auto,
    )
    .unwrap_err();
    match err {
        ResolveError::UnsupportedCombo(msg) => {
            assert!(
                msg.contains("q8_g64"),
                "expected K codec named in msg: {msg}"
            );
            assert!(msg.contains("tq4"), "expected V codec named in msg: {msg}");
        }
        other => panic!("expected UnsupportedCombo, got {other:?}"),
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn healthy_k8v4_coercion_from_auto_k8v8_plus_tq4() {
    // K8V8 → decompose gives (Q8G128, Q8G128). User asks --ctk auto --ctv tq4
    // → K stays Q8G128, V becomes Tq4. combo_to_kv_quant coerces to K8V4.
    let kq = resolve(
        spec(CacheType::Auto, CacheType::Tq4),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(kq, KvQuant::K8V4);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn qwen_moe_all_auto_k8v8_accepted() {
    let kq = resolve(
        spec(CacheType::Auto, CacheType::Auto),
        ctx("Qwen3_5MoeForConditionalGeneration", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(kq, KvQuant::K8V8);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn qwen_moe_low_k_bits_rejected_post_decompose() {
    // Hypothetical auto: Mixed{k_bits:4, v_bits:4, group=64}. This proves
    // the post-decompose §D6.4 re-check fires even if resolve_default
    // ever returned a bad value.
    let bad_auto = KvQuant::Mixed {
        k_bits: 4,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let err = resolve(
        spec(CacheType::Auto, CacheType::Auto),
        ctx("Qwen3_5MoeForConditionalGeneration", Some(128)),
        bad_auto,
    )
    .unwrap_err();
    assert_eq!(err, ResolveError::QwenMoeKBitsTooLow(4));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn head_dim_80_q4_g64_rejected_d6_1() {
    // head_dim=80; 80 % 64 != 0 → GroupSizeNotDivisible.
    let err = resolve(
        spec(CacheType::Auto, CacheType::Q4G64),
        ctx("Qwen3ForCausalLM", Some(80)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    match err {
        ResolveError::GroupSizeNotDivisible {
            head_dim,
            group_size,
        } => {
            assert_eq!(head_dim, 80);
            assert_eq!(group_size, 64);
        }
        other => panic!("expected GroupSizeNotDivisible, got {other:?}"),
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn mlx_bit_packing_violation_q3_g64_on_unfriendly_head_dim() {
    // q3_g64: bits=3, group=64. MLX bit-packing: head_dim % (32/3 = 10) == 0.
    // group_size=64 must also divide head_dim (D6.1).
    // Pick head_dim=64: 64 % 64 == 0 (D6.1 OK), 64 % 10 == 4 → D6.2 fires.
    let err = resolve(
        spec(CacheType::Auto, CacheType::Q3G64),
        ctx("Qwen3ForCausalLM", Some(64)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    match err {
        ResolveError::MlxBitPackingViolation { head_dim, bits } => {
            assert_eq!(head_dim, 64);
            assert_eq!(bits, 3);
        }
        other => panic!("expected MlxBitPackingViolation, got {other:?}"),
    }
}

// ── Gemma family + Mixed is now ACCEPTED (dequant-before-share) ─────

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma4_mixed_accepted_via_cache_type_spec() {
    // Gemma4 with an explicit Mixed combo (q8_g128 K + q4_g64 V) now
    // resolves successfully — the shared-KV path dequantises before share.
    // head_dim=256 for e4b.
    let kq = resolve(
        spec(CacheType::Q8G128, CacheType::Q4G64),
        ctx("Gemma4ForConditionalGeneration", Some(256)),
        KvQuant::K8V8,
    )
    .expect("Gemma4 + Mixed must resolve");
    assert_eq!(
        kq,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 128,
            v_group_size: 64,
        }
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma3_mixed_accepted_via_cache_type_spec() {
    // Gemma3 + Mixed also resolves (same shared-KV wrapper supports it).
    let kq = resolve(
        spec(CacheType::Q8G128, CacheType::Q4G64),
        ctx("Gemma3ForConditionalGeneration", Some(256)),
        KvQuant::K8V8,
    )
    .expect("Gemma3 + Mixed must resolve");
    assert!(matches!(kq, KvQuant::Mixed { .. }));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gemma4_k8v8_accepted() {
    // Non-Mixed quant on Gemma4 must still work.
    let kq = resolve(
        spec(CacheType::Auto, CacheType::Auto),
        ctx("Gemma4ForConditionalGeneration", Some(256)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(kq, KvQuant::K8V8);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn non_gemma_mixed_accepted() {
    // Bonsai (Qwen3ForCausalLM) with Mixed is fine — no shared-KV.
    let kq = resolve(
        spec(CacheType::Q8G64, CacheType::Q4G64),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(
        kq,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn validate_resolved_gemma4_mixed_passes() {
    // validate_resolved no longer rejects Gemma4 + Mixed — the
    // shared-KV wrapper dequantises before share. (Exercised by the preset
    // `--kv-quant mixed` path in parse.rs.)
    let mixed = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 128,
        v_group_size: 64,
    };
    validate_resolved("Gemma4ForConditionalGeneration", &mixed)
        .expect("Gemma4 + Mixed must pass validate_resolved");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn validate_resolved_gemma4_k8v8_passes() {
    // K8V8 on Gemma4 must still pass validate_resolved.
    validate_resolved("Gemma4ForConditionalGeneration", &KvQuant::K8V8).unwrap();
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn validate_resolved_qwen_moe_tsym4_rejected() {
    // Arch guard: TurboSym4 (symmetric WHT-4 K + tq4 V) on Qwen MoE
    // is the PPL-218→8641 disaster path (CLAUDE.md hard rule 6). Rejected by
    // `validate_resolved` with `QwenMoeKBitsTooLow(4)` (same error class as
    // the existing Mixed K<8 rejection — uniform exit code + hint surface).
    let err =
        validate_resolved("Qwen3_5MoeForConditionalGeneration", &KvQuant::TurboSym4).unwrap_err();
    assert_eq!(err, ResolveError::QwenMoeKBitsTooLow(4));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — we want a panic if the guard did not fire, so the assertion is the safety net"
)]
fn validate_resolved_qwen3vl_moe_tsym4_rejected() {
    // Qwen3VLMoeForConditionalGeneration is also a
    // Qwen sparse-MoE and must be covered by the is_qwen_moe guard.
    // TurboSym4 on VL-MoE is the same PPL-disaster path as on text-only MoE.
    let err =
        validate_resolved("Qwen3VLMoeForConditionalGeneration", &KvQuant::TurboSym4).unwrap_err();
    assert_eq!(err, ResolveError::QwenMoeKBitsTooLow(4));
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn validate_resolved_gemma4_tsym4_passes() {
    // TurboSym4 is allowed on non-Qwen-MoE archs. The arch guard
    // is Qwen-MoE-specific; Gemma4 / Qwen3 dense pass through cleanly.
    validate_resolved("Gemma4ForConditionalGeneration", &KvQuant::TurboSym4).unwrap();
    validate_resolved("Qwen3ForCausalLM", &KvQuant::TurboSym4).unwrap();
}

// Contract A.y — PlanarK is K-side 4-bit rotation and MUST be
// rejected on Qwen MoE. Covers BOTH text-only (`Qwen3_5MoeForConditionalGeneration`)
// and vision-language (`Qwen3VLMoeForConditionalGeneration`) MoE arch strings.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — we want a panic if the guard did not fire, so the assertion is the safety net"
)]
fn validate_resolved_qwen_moe_planar_k_rejected() {
    // text-only MoE
    let err =
        validate_resolved("Qwen3_5MoeForConditionalGeneration", &KvQuant::PlanarK).unwrap_err();
    assert_eq!(err, ResolveError::QwenMoePlanarKRejected);
    // vision-language MoE — same guard via is_qwen_moe helper.
    let err =
        validate_resolved("Qwen3VLMoeForConditionalGeneration", &KvQuant::PlanarK).unwrap_err();
    assert_eq!(err, ResolveError::QwenMoePlanarKRejected);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn validate_resolved_gemma4_planar_k_passes() {
    // PlanarK is allowed on non-Qwen-MoE archs (Gemma4, Bonsai).
    validate_resolved("Gemma4ForConditionalGeneration", &KvQuant::PlanarK).unwrap();
    validate_resolved("Qwen3ForCausalLM", &KvQuant::PlanarK).unwrap();
}

// Explicit per-side spec (`--ctk planar_k4 --ctv bf16`) on Qwen MoE
// also flows through `validate_resolved` post-decompose and must hit the guard.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — we want a panic if the guard did not fire, so the assertion is the safety net"
)]
fn qwen_moe_planar_k4_per_side_spec_rejected() {
    let err = resolve(
        spec(CacheType::PlanarK4, CacheType::Bf16),
        ctx("Qwen3_5MoeForConditionalGeneration", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ResolveError::QwenMoePlanarKRejected,
        "--ctk planar_k4 on Qwen MoE must hit the PlanarK K-disaster guard"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn validate_resolved_bonsai_mixed_passes() {
    // Qwen3ForCausalLM (Bonsai) with Mixed must pass — not a shared-KV arch.
    let mixed = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    validate_resolved("Qwen3ForCausalLM", &mixed).unwrap();
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn combo_q8g128_q4g64_yields_mixed() {
    // User-explicit Mixed combo (no auto involvement).
    let kq = resolve(
        spec(CacheType::Q8G128, CacheType::Q4G64),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap();
    assert_eq!(
        kq,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 128,
            v_group_size: 64,
        }
    );
}

// ── Rotor4 (4-bit Clifford rotor V) resolver tests ────────────────────────────

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rotor4_explicit_resolves_to_kv_quant_rotor4() {
    // (Q8G128, Rotor4) → KvQuant::Rotor4.
    // head_dim=128 satisfies Q8G128 (128 % 128 == 0) and Rotor4 (non-affine,
    // no group-size constraint).
    let kq = resolve(
        spec(CacheType::Q8G128, CacheType::Rotor4),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V4,
    )
    .unwrap();
    assert_eq!(kq, KvQuant::Rotor4);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rotor4_k_side_rejected() {
    // Rotor4 is V-only — reject on K side.
    let err = resolve(
        spec(CacheType::Rotor4, CacheType::Auto),
        ctx("Qwen3ForCausalLM", Some(128)),
        KvQuant::K8V8,
    )
    .unwrap_err();
    assert!(
        matches!(err, ResolveError::KSideRotationCodec(tag) if tag == "rotor_v_4"),
        "got {err:?}"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rotor4_parse_round_trip() {
    // "rotor_v_4" and "rotor4" both parse to CacheType::Rotor4.
    assert_eq!(parse("rotor_v_4").ok(), Some(CacheType::Rotor4));
    assert_eq!(parse("rotor4").ok(), Some(CacheType::Rotor4));
    assert_eq!(CacheType::Rotor4.tag(), "rotor_v_4");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rotor4_non_q8g128_k_rejected() {
    // Rotor4 with non-Q8G128 K is rejected.
    let err = combo_to_kv_quant(CacheType::Q8G64, CacheType::Rotor4).unwrap_err();
    assert!(
        matches!(err, ResolveError::UnsupportedCombo(_)),
        "expected UnsupportedCombo, got {err:?}"
    );
}

// ── K-side IsoQuant tests ─────────────────────────────────────────────────────

/// Iso3Sym on Qwen MoE must fire the A.y guard
/// (QwenMoeIsoKRejected with the offending variant name).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the QwenMoeIsoKRejected arm only; wildcard panics on guard drift"
)]
fn validate_resolved_qwen_moe_iso3_sym_rejected() {
    let err =
        validate_resolved("Qwen3_5MoeForConditionalGeneration", &KvQuant::Iso3Sym).unwrap_err();
    match err {
        ResolveError::QwenMoeIsoKRejected { ref variant } => {
            assert_eq!(variant, "iso3_sym", "guard must name the offending variant");
        }
        other => panic!("expected QwenMoeIsoKRejected, got {other:?}"),
    }
    // vision-language MoE — same guard via is_qwen_moe helper.
    let err =
        validate_resolved("Qwen3VLMoeForConditionalGeneration", &KvQuant::Iso3Sym).unwrap_err();
    assert!(matches!(err, ResolveError::QwenMoeIsoKRejected { .. }));
}

/// Iso4Sym / IsoKOnly3 / IsoKOnly4 — same A.y guard fires.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire on every iso K-side variant"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the QwenMoeIsoKRejected arm only"
)]
fn validate_resolved_qwen_moe_all_iso_k_side_rejected() {
    for kq in [KvQuant::Iso4Sym, KvQuant::IsoKOnly3, KvQuant::IsoKOnly4] {
        let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
        let ResolveError::QwenMoeIsoKRejected { ref variant } = err else {
            panic!("expected QwenMoeIsoKRejected for {kq:?}, got {err:?}");
        };
        let expected = kq.to_string();
        assert_eq!(variant, &expected, "variant name mismatch for {kq:?}");
    }
}

/// Non-MoE archs (Gemma4, Bonsai) accept all four iso K-side
/// variants (A.y guard does not fire).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts the guard passes on non-Qwen-MoE — unwrap surfaces a wrong-arch leak"
)]
fn validate_resolved_non_moe_iso_k_side_passes() {
    for kq in [
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
    ] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq).unwrap();
        validate_resolved("Qwen3ForCausalLM", &kq).unwrap();
    }
}

/// Metal-vs-CPU classification in `validate_resolved` (warn-only).
///
/// CPU-hot-path iso / rotor V-only codecs resolve `Ok` — the classifier emits
/// a warn but never rejects. Metal codecs also resolve `Ok`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts Ok for all paths — unwrap surfaces an unexpected reject"
)]
fn validate_resolved_cpu_codec_classification() {
    // CPU-hot-path codecs (iso/rotor V-only families): warn-and-proceed,
    // never rejected.
    for kq in [
        KvQuant::Iso3,
        KvQuant::Rotor3,
        KvQuant::Iso4,
        KvQuant::Rotor4,
    ] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq)
            .unwrap_or_else(|e| panic!("CPU-hot-path codec must resolve Ok for {kq}: {e}"));
    }

    // Metal codecs resolve Ok.
    for kq in [KvQuant::K8V4, KvQuant::Planar, KvQuant::RotKTq4V] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq)
            .unwrap_or_else(|e| panic!("Metal codec must resolve Ok for {kq}: {e}"));
    }

    // K-only iso codecs dispatch the iso K MSL kernel every decode step — they
    // are Metal (cpu_hot_path_reason returns None) and must resolve Ok.
    for kq in [KvQuant::IsoKOnly3, KvQuant::IsoKOnly4] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq)
            .unwrap_or_else(|e| panic!("K-only iso codec must resolve Ok for {kq}: {e}"));
    }

    // K-only rotor codec verdict is QJL-dependent; in either case
    // validate_resolved must return Ok (no reject path).
    for kq in [KvQuant::RotorKOnly3, KvQuant::RotorKOnly4] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq)
            .unwrap_or_else(|e| panic!("K-only rotor codec must resolve Ok for {kq}: {e}"));
    }
}

/// Iso3Sym combo_to_kv_quant: (IsoK3, Iso3) maps to Iso3Sym.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: combo returns Ok by construction for the documented pair"
)]
fn combo_to_kv_quant_iso3_sym() {
    let kq = combo_to_kv_quant(CacheType::IsoK3, CacheType::Iso3).unwrap();
    assert_eq!(kq, KvQuant::Iso3Sym);
    let kq = combo_to_kv_quant(CacheType::IsoK4, CacheType::Iso4).unwrap();
    assert_eq!(kq, KvQuant::Iso4Sym);
}

/// IsoK3 + Bf16 V → IsoKOnly3.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: combo returns Ok by construction"
)]
fn combo_to_kv_quant_iso_k_only() {
    let kq = combo_to_kv_quant(CacheType::IsoK3, CacheType::Bf16).unwrap();
    assert_eq!(kq, KvQuant::IsoKOnly3);
    let kq = combo_to_kv_quant(CacheType::IsoK4, CacheType::Bf16).unwrap();
    assert_eq!(kq, KvQuant::IsoKOnly4);
}

/// IsoK3 paired with a non-iso V is rejected.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire"
)]
fn combo_to_kv_quant_iso_k3_wrong_v_rejected() {
    let err = combo_to_kv_quant(CacheType::IsoK3, CacheType::Tq4).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
}

/// IsoK3 / IsoK4 on V side is rejected (K-side codecs only).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire"
)]
fn iso_k_v_side_rejected() {
    let err = combo_to_kv_quant(CacheType::Q8G128, CacheType::IsoK3).unwrap_err();
    assert!(
        matches!(err, ResolveError::UnsupportedCombo(_)),
        "got {err:?}"
    );
    let err = combo_to_kv_quant(CacheType::Q8G128, CacheType::IsoK4).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
}

/// KvQuant FromStr roundtrip for the iso K-side spellings.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: FromStr returns Ok by construction for the documented spellings"
)]
fn kv_quant_parse_iso_variants() {
    use std::str::FromStr;
    assert_eq!(KvQuant::from_str("iso3_sym").unwrap(), KvQuant::Iso3Sym);
    assert_eq!(KvQuant::from_str("iso4_sym").unwrap(), KvQuant::Iso4Sym);
    assert_eq!(KvQuant::from_str("k_iso3").unwrap(), KvQuant::IsoKOnly3);
    assert_eq!(KvQuant::from_str("k_iso4").unwrap(), KvQuant::IsoKOnly4);
    // Display round-trip.
    assert_eq!(format!("{}", KvQuant::Iso3Sym), "iso3_sym");
    assert_eq!(format!("{}", KvQuant::IsoKOnly4), "k_iso4");
}

// ── Rotor K-side cache_type tests ────────────────────────────────────────────

/// A.y Qwen MoE guard for every rotor K-side KvQuant variant.
/// Asserts BOTH discriminator AND `variant` payload string (LOW-bug guard: must match both).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire on every rotor K-side variant"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the QwenMoeRotorKRejected arm only"
)]
fn validate_resolved_qwen_moe_all_rotor_k_side_rejected() {
    for kq in [
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
    ] {
        let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
        let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
            panic!("expected QwenMoeRotorKRejected for {kq:?}, got {err:?}");
        };
        let expected = kq.to_string();
        assert_eq!(variant, &expected, "variant name mismatch for {kq:?}");
    }
    // VL Qwen MoE: same guard via is_qwen_moe.
    let err =
        validate_resolved("Qwen3VLMoeForConditionalGeneration", &KvQuant::Rotor3Sym).unwrap_err();
    assert!(matches!(err, ResolveError::QwenMoeRotorKRejected { .. }));
}

/// A.y guard for the 2 internal CacheType K-side variants
/// (RotorK3 / RotorK4) via combo_to_kv_quant → validate_resolved pipeline.
/// Asserts both error discriminator and variant payload.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — combo + guard must fire on Qwen MoE"
)]
fn validate_resolved_qwen_moe_rotor_cache_type_rejected() {
    // (RotorK3, Rotor3) → Rotor3Sym → rejected on Qwen MoE.
    let kq = combo_to_kv_quant(CacheType::RotorK3, CacheType::Rotor3).unwrap();
    let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
    let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
        panic!("expected QwenMoeRotorKRejected for RotorK3+Rotor3, got {err:?}");
    };
    assert_eq!(variant, "rotor3_sym");

    // (RotorK4, Rotor4) → Rotor4Sym → rejected.
    let kq = combo_to_kv_quant(CacheType::RotorK4, CacheType::Rotor4).unwrap();
    let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
    let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
        panic!("expected QwenMoeRotorKRejected for RotorK4+Rotor4, got {err:?}");
    };
    assert_eq!(variant, "rotor4_sym");

    // (RotorK3, Bf16) → RotorKOnly3 → rejected on Qwen MoE.
    let kq = combo_to_kv_quant(CacheType::RotorK3, CacheType::Bf16).unwrap();
    let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
    let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
        panic!("expected QwenMoeRotorKRejected for (RotorK3, Bf16), got {err:?}");
    };
    assert_eq!(variant, "k_rotor3");

    // (RotorK4, Bf16) → RotorKOnly4 → rejected on Qwen MoE.
    let kq = combo_to_kv_quant(CacheType::RotorK4, CacheType::Bf16).unwrap();
    let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
    let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
        panic!("expected QwenMoeRotorKRejected for (RotorK4, Bf16), got {err:?}");
    };
    assert_eq!(variant, "k_rotor4");
}

/// Non-MoE archs accept all four rotor K-side variants.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts the guard passes on non-Qwen-MoE — unwrap surfaces a wrong-arch leak"
)]
fn validate_resolved_non_moe_rotor_k_side_passes() {
    for kq in [
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
    ] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq).unwrap();
        validate_resolved("Qwen3ForCausalLM", &kq).unwrap();
        // Dense Qwen3.5 shares a loader and an `Architecture` variant with the
        // sparse-MoE path, so this string is a value `arch_class()` can now
        // return. The measured PPL disaster is a sparse-MoE result; a dense
        // model keeps these codecs.
        validate_resolved("Qwen3_5ForConditionalGeneration", &kq).unwrap();
    }
}

/// The two Qwen3.5 arch strings produce OPPOSITE verdicts for every codec the
/// guard exists to reject.
///
/// Weights-free proof that which string reaches `validate_resolved` decides
/// whether the guard fires at all — so `Architecture::arch_class()` returning
/// the declared name instead of the resolved one is a correctness bug, not a
/// labelling nit. The seam that feeds the resolved name in
/// (`Architecture::generate_greedy`) can only be exercised with a real
/// snapshot; see `tests/resolved_arch_class.rs` and docs/TESTING.md.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts the dense arm passes — unwrap surfaces an over-broad refusal"
)]
fn validate_resolved_qwen3_5_dense_and_moe_strings_diverge() {
    for kq in [
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::PlanarK,
        KvQuant::TurboSym3,
        KvQuant::TurboSym4,
    ] {
        assert!(
            validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).is_err(),
            "{kq} must be rejected on sparse Qwen3.5 MoE"
        );
        validate_resolved("Qwen3_5ForConditionalGeneration", &kq).unwrap();
    }
}

/// combo_to_kv_quant: (RotorK3, Rotor3) maps to Rotor3Sym; etc.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: combo returns Ok by construction"
)]
fn combo_to_kv_quant_rotor3_sym() {
    let kq = combo_to_kv_quant(CacheType::RotorK3, CacheType::Rotor3).unwrap();
    assert_eq!(kq, KvQuant::Rotor3Sym);
    let kq = combo_to_kv_quant(CacheType::RotorK4, CacheType::Rotor4).unwrap();
    assert_eq!(kq, KvQuant::Rotor4Sym);
}

/// (RotorK3, Bf16) → RotorKOnly3; (RotorK4, Bf16) → RotorKOnly4.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: combo returns Ok by construction"
)]
fn combo_to_kv_quant_rotor_k_only() {
    let kq = combo_to_kv_quant(CacheType::RotorK3, CacheType::Bf16).unwrap();
    assert_eq!(kq, KvQuant::RotorKOnly3);
    let kq = combo_to_kv_quant(CacheType::RotorK4, CacheType::Bf16).unwrap();
    assert_eq!(kq, KvQuant::RotorKOnly4);
}

/// RotorK3 paired with a non-rotor V is rejected.
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn combo_to_kv_quant_rotor_k3_wrong_v_rejected() {
    let err = combo_to_kv_quant(CacheType::RotorK3, CacheType::Tq4).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
}

/// RotorK3 / RotorK4 on V side is rejected (K-side codecs only).
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn rotor_k_v_side_rejected() {
    let err = combo_to_kv_quant(CacheType::Q8G128, CacheType::RotorK3).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
    let err = combo_to_kv_quant(CacheType::Q8G128, CacheType::RotorK4).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
}

/// KvQuant FromStr roundtrip + Display for rotor K-side variants.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: FromStr returns Ok by construction"
)]
fn kv_quant_parse_rotor_variants() {
    use std::str::FromStr;
    assert_eq!(KvQuant::from_str("rotor3_sym").unwrap(), KvQuant::Rotor3Sym);
    assert_eq!(KvQuant::from_str("rotor4_sym").unwrap(), KvQuant::Rotor4Sym);
    assert_eq!(KvQuant::from_str("k_rotor3").unwrap(), KvQuant::RotorKOnly3);
    assert_eq!(KvQuant::from_str("k_rotor4").unwrap(), KvQuant::RotorKOnly4);
    assert_eq!(format!("{}", KvQuant::Rotor3Sym), "rotor3_sym");
    assert_eq!(format!("{}", KvQuant::RotorKOnly4), "k_rotor4");
}

// ── TurboSym3 cache_type tests ───────────────────────────────────────────────

/// (TurboSym3, TurboSym3) → KvQuant::TurboSym3.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: combo returns Ok by construction"
)]
fn combo_to_kv_quant_tsym3_symmetric() {
    let kq = combo_to_kv_quant(CacheType::TurboSym3, CacheType::TurboSym3).unwrap();
    assert_eq!(kq, KvQuant::TurboSym3);
}

/// TurboSym3 K-side with non-TurboSym3 V is rejected.
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn combo_to_kv_quant_tsym3_wrong_v_rejected() {
    let err = combo_to_kv_quant(CacheType::TurboSym3, CacheType::Tq4).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
    let err = combo_to_kv_quant(CacheType::TurboSym3, CacheType::Bf16).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
}

/// TurboSym3 on V side with non-TurboSym3 K is rejected.
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn combo_to_kv_quant_tsym3_wrong_k_rejected() {
    let err = combo_to_kv_quant(CacheType::Q8G128, CacheType::TurboSym3).unwrap_err();
    assert!(matches!(err, ResolveError::UnsupportedCombo(_)));
}

/// A.y Qwen MoE guard for TurboSym3 (K-side 3-bit disaster).
/// Asserts BOTH discriminator AND `variant` payload string (LOW-bug guard: must match both).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire on Qwen MoE"
)]
fn validate_resolved_qwen_moe_tsym3_rejected() {
    let err =
        validate_resolved("Qwen3_5MoeForConditionalGeneration", &KvQuant::TurboSym3).unwrap_err();
    let ResolveError::QwenMoeTurboKRejected { ref variant } = err else {
        panic!("expected QwenMoeTurboKRejected for TurboSym3, got {err:?}");
    };
    assert_eq!(variant, "tsym3", "variant name mismatch");
    // VL Qwen MoE: same guard via is_qwen_moe.
    let err =
        validate_resolved("Qwen3VLMoeForConditionalGeneration", &KvQuant::TurboSym3).unwrap_err();
    assert!(matches!(err, ResolveError::QwenMoeTurboKRejected { .. }));
}

/// Non-MoE archs accept TurboSym3.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts the guard passes on non-Qwen-MoE — unwrap surfaces a wrong-arch leak"
)]
fn validate_resolved_non_moe_tsym3_passes() {
    validate_resolved("Gemma4ForConditionalGeneration", &KvQuant::TurboSym3).unwrap();
    validate_resolved("Qwen3ForCausalLM", &KvQuant::TurboSym3).unwrap();
}

/// KvQuant FromStr roundtrip + Display for TurboSym3.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: FromStr returns Ok by construction"
)]
fn kv_quant_parse_tsym3() {
    use std::str::FromStr;
    assert_eq!(KvQuant::from_str("tsym3").unwrap(), KvQuant::TurboSym3);
    assert_eq!(format!("{}", KvQuant::TurboSym3), "tsym3");
}

/// parse("tsym3") returns CacheType::TurboSym3.
#[test]
#[allow(clippy::unwrap_used, reason = "test scaffolding")]
fn parse_tsym3() {
    assert_eq!(parse("tsym3").unwrap(), CacheType::TurboSym3);
}

/// tag() and all() include TurboSym3.
#[test]
fn tsym3_in_all_and_tag() {
    assert_eq!(CacheType::TurboSym3.tag(), "tsym3");
    assert!(
        CacheType::all().contains(&CacheType::TurboSym3),
        "TurboSym3 must be in all()"
    );
}

// ── A.y guard verification for all fused-QK K-side KvQuant ──────────────────
//
// Every K-side ≤4-bit KvQuant that has a fused-QK kernel must be rejected on
// Qwen MoE (`Qwen3_5MoeForConditionalGeneration`) by `validate_resolved`.
//
// Verifies the A.y guard contract that the kernel dispatch table assumes:
// callers rely on `validate_resolved` having fired at session start to ensure
// no K-side ≤4-bit codec reaches the Qwen MoE forward pass.

/// A.y guard — TurboSym3 on Qwen MoE → QwenMoeTurboKRejected.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the QwenMoeTurboKRejected arm only"
)]
fn validate_resolved_fused_qk_ay_guard_turbosym3() {
    let err =
        validate_resolved("Qwen3_5MoeForConditionalGeneration", &KvQuant::TurboSym3).unwrap_err();
    match err {
        ResolveError::QwenMoeTurboKRejected { ref variant } => {
            assert_eq!(variant, "tsym3", "guard must name the offending variant");
        }
        other => panic!("expected QwenMoeTurboKRejected for TurboSym3, got {other:?}"),
    }
}

/// A.y guard — TurboSym4 on Qwen MoE → QwenMoeKBitsTooLow(4).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire"
)]
fn validate_resolved_fused_qk_ay_guard_turbosym4() {
    let err =
        validate_resolved("Qwen3_5MoeForConditionalGeneration", &KvQuant::TurboSym4).unwrap_err();
    assert!(
        matches!(err, ResolveError::QwenMoeKBitsTooLow(4)),
        "expected QwenMoeKBitsTooLow(4) for TurboSym4 on Qwen MoE, got {err:?}"
    );
}

/// A.y guard — IsoSym3 and IsoSym4 on Qwen MoE → QwenMoeIsoKRejected.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire on both iso sym variants"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the QwenMoeIsoKRejected arm only"
)]
fn validate_resolved_fused_qk_ay_guard_iso_sym() {
    for (kq, expected_variant) in [
        (KvQuant::Iso3Sym, "iso3_sym"),
        (KvQuant::Iso4Sym, "iso4_sym"),
    ] {
        let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
        let ResolveError::QwenMoeIsoKRejected { ref variant } = err else {
            panic!("expected QwenMoeIsoKRejected for {kq:?}, got {err:?}");
        };
        assert_eq!(
            variant, expected_variant,
            "variant name mismatch for {kq:?}"
        );
    }
}

/// A.y guard — RotorSym3 and RotorSym4 on Qwen MoE → QwenMoeRotorKRejected.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire on both rotor sym variants"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts the QwenMoeRotorKRejected arm only"
)]
fn validate_resolved_fused_qk_ay_guard_rotor_sym() {
    for (kq, expected_variant) in [
        (KvQuant::Rotor3Sym, "rotor3_sym"),
        (KvQuant::Rotor4Sym, "rotor4_sym"),
    ] {
        let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
        let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
            panic!("expected QwenMoeRotorKRejected for {kq:?}, got {err:?}");
        };
        assert_eq!(
            variant, expected_variant,
            "variant name mismatch for {kq:?}"
        );
    }
}

// ── RotorK{3,4}Asym tests ────────────────────────────────────────────────────

/// KvQuant Display + FromStr round-trip across the accepted
/// affine V codecs (single representative per (bits, group) pair).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: FromStr returns Ok by construction for canonical tags"
)]
fn kv_quant_parse_rotor_k_asym_round_trip() {
    use std::str::FromStr;
    let cases: &[(u8, u16)] = &[(4, 128), (4, 64), (4, 32), (3, 64), (2, 64)];
    for &(vb, vg) in cases {
        for ctor_tag in [("rotor_k_3_asym", true), ("rotor_k_4_asym", false)] {
            let kq = if ctor_tag.1 {
                KvQuant::RotorK3Asym {
                    v_bits: vb,
                    v_group_size: vg,
                }
            } else {
                KvQuant::RotorK4Asym {
                    v_bits: vb,
                    v_group_size: vg,
                }
            };
            let tag = format!("{kq}");
            let expected = format!("{}_v{}_g{}", ctor_tag.0, vb, vg);
            assert_eq!(tag, expected, "Display mismatch for {kq:?}");
            assert_eq!(KvQuant::from_str(&tag).unwrap(), kq, "FromStr round-trip");
        }
    }
}

/// FromStr rejects (`v_bits`, `v_group_size`) tuples that have
/// no affine V codec mapping.
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn kv_quant_parse_rotor_k_asym_rejects_unknown_v() {
    use std::str::FromStr;
    // (v_bits=8, *) — TurboQuant has no 8-bit path; rejected at construction.
    let err = KvQuant::from_str("rotor_k_3_asym_v8_g128").unwrap_err();
    assert!(
        matches!(
            err,
            rmlx_kv_quant::KvQuantParseError::InvalidRotorKAsym { .. }
        ),
        "expected InvalidRotorKAsym, got {err:?}"
    );
    // (v_bits=4, v_group_size=256) — not a canonical TurboQuant V tuple.
    let err = KvQuant::from_str("rotor_k_3_asym_v4_g256").unwrap_err();
    assert!(
        matches!(
            err,
            rmlx_kv_quant::KvQuantParseError::InvalidRotorKAsym { .. }
        ),
        "expected InvalidRotorKAsym, got {err:?}"
    );
    // (v_bits=3, v_group_size=128) — 3-bit V requires group=64.
    let err = KvQuant::from_str("rotor_k_3_asym_v3_g128").unwrap_err();
    assert!(
        matches!(
            err,
            rmlx_kv_quant::KvQuantParseError::InvalidRotorKAsym { .. }
        ),
        "expected InvalidRotorKAsym for (3, 128), got {err:?}"
    );
}

/// K4 mirror of `kv_quant_parse_rotor_k_asym_rejects_unknown_v`.
/// Covers the same (v_bits, v_group_size) rejection invariants from the K4 entry
/// point so a regression on the K4 parse path cannot hide behind the K3 case.
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn kv_quant_parse_rotor_k_4_asym_rejects_unknown_v() {
    use std::str::FromStr;
    // (v_bits=8, *) — TurboQuant has no 8-bit path.
    let err = KvQuant::from_str("rotor_k_4_asym_v8_g128").unwrap_err();
    assert!(
        matches!(
            err,
            rmlx_kv_quant::KvQuantParseError::InvalidRotorKAsym { .. }
        ),
        "expected InvalidRotorKAsym, got {err:?}"
    );
    // (v_bits=4, v_group_size=256) — not canonical.
    let err = KvQuant::from_str("rotor_k_4_asym_v4_g256").unwrap_err();
    assert!(
        matches!(
            err,
            rmlx_kv_quant::KvQuantParseError::InvalidRotorKAsym { .. }
        ),
        "expected InvalidRotorKAsym, got {err:?}"
    );
    // (v_bits=3, v_group_size=128) — 3-bit V requires group=64.
    let err = KvQuant::from_str("rotor_k_4_asym_v3_g128").unwrap_err();
    assert!(
        matches!(
            err,
            rmlx_kv_quant::KvQuantParseError::InvalidRotorKAsym { .. }
        ),
        "expected InvalidRotorKAsym for (3, 128), got {err:?}"
    );
}

/// `(RotorK3, Q4G128)` → `RotorK3Asym { v_bits: 4, v_group_size: 128 }`.
#[test]
#[allow(clippy::unwrap_used, reason = "test scaffolding")]
fn combo_to_kv_quant_rotor_k3_asym_q4_g128() {
    let kq = combo_to_kv_quant(CacheType::RotorK3, CacheType::Q4G128).unwrap();
    assert_eq!(
        kq,
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        }
    );
}

/// `(RotorK3, Q8G128)` is rejected (TurboQuant V has no 8-bit path).
#[test]
#[allow(clippy::unwrap_used, reason = "test intentionally calls unwrap_err")]
fn combo_to_kv_quant_rotor_k3_rejects_q8() {
    let err = combo_to_kv_quant(CacheType::RotorK3, CacheType::Q8G128).unwrap_err();
    assert!(
        matches!(err, ResolveError::UnsupportedCombo(_)),
        "expected UnsupportedCombo for (RotorK3, Q8G128), got {err:?}"
    );
}

/// `(RotorK4, Q4G64)` → `RotorK4Asym { v_bits: 4, v_group_size: 64 }`.
#[test]
#[allow(clippy::unwrap_used, reason = "test scaffolding")]
fn combo_to_kv_quant_rotor_k4_asym_q4_g64() {
    let kq = combo_to_kv_quant(CacheType::RotorK4, CacheType::Q4G64).unwrap();
    assert_eq!(
        kq,
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        }
    );
}

/// RotorK*Asym variants are A.y-guarded on Qwen MoE (rejected with
/// the same `QwenMoeRotorKRejected` error string as the sym/k-only siblings).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test intentionally calls unwrap_err — guard must fire on all asym v codecs"
)]
fn validate_resolved_qwen_moe_rotor_k_asym_rejected() {
    let cases = [
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::RotorK3Asym {
            v_bits: 2,
            v_group_size: 64,
        },
    ];
    for kq in cases {
        let err = validate_resolved("Qwen3_5MoeForConditionalGeneration", &kq).unwrap_err();
        let ResolveError::QwenMoeRotorKRejected { ref variant } = err else {
            panic!("expected QwenMoeRotorKRejected for {kq:?}, got {err:?}");
        };
        let expected = format!("{kq}");
        assert_eq!(variant, &expected);
    }
    // VL Qwen MoE: same guard.
    let kq = KvQuant::RotorK3Asym {
        v_bits: 4,
        v_group_size: 128,
    };
    let err = validate_resolved("Qwen3VLMoeForConditionalGeneration", &kq).unwrap_err();
    assert!(matches!(err, ResolveError::QwenMoeRotorKRejected { .. }));
}

/// Non-MoE archs accept all asymmetric rotor-K variants.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts the guard passes on non-Qwen-MoE — unwrap surfaces a wrong-arch leak"
)]
fn validate_resolved_non_moe_rotor_k_asym_passes() {
    for kq in [
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
    ] {
        validate_resolved("Gemma4ForConditionalGeneration", &kq).unwrap();
        validate_resolved("Qwen3ForCausalLM", &kq).unwrap();
    }
}

/// `decompose_auto(RotorK3Asym { … })` round-trips back through
/// `combo_to_kv_quant`.
#[test]
#[allow(clippy::unwrap_used, reason = "test scaffolding: round-trip is total")]
fn decompose_auto_rotor_k_asym_round_trip() {
    let original = KvQuant::RotorK3Asym {
        v_bits: 4,
        v_group_size: 128,
    };
    let (k, v) = decompose_auto(original);
    assert_eq!(k, CacheType::RotorK3);
    assert_eq!(v, CacheType::Q4G128);
    assert_eq!(combo_to_kv_quant(k, v).unwrap(), original);

    let original = KvQuant::RotorK4Asym {
        v_bits: 4,
        v_group_size: 64,
    };
    let (k, v) = decompose_auto(original);
    assert_eq!(k, CacheType::RotorK4);
    assert_eq!(v, CacheType::Q4G64);
    assert_eq!(combo_to_kv_quant(k, v).unwrap(), original);
}
