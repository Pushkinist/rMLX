use std::str::FromStr;

use super::*;
use rmlx_models::kv_cache::{CacheType, CacheTypeSpec};

// ── parse_cache_type ─────────────────────────────────────────────────────

#[test]
fn parse_cache_type_canonical_tags_round_trip() {
    let canonical = [
        ("auto", CacheType::Auto),
        ("bf16", CacheType::Bf16),
        ("q8_g128", CacheType::Q8G128),
        ("q8_g64", CacheType::Q8G64),
        ("q8_g32", CacheType::Q8G32),
        ("q6_g64", CacheType::Q6G64),
        ("q5_g64", CacheType::Q5G64),
        ("q4_g128", CacheType::Q4G128),
        ("q4_g64", CacheType::Q4G64),
        ("q4_g32", CacheType::Q4G32),
        ("q3_g64", CacheType::Q3G64),
        ("q2_g64", CacheType::Q2G64),
        ("tq4", CacheType::Tq4),
        ("planar4", CacheType::Planar4),
    ];
    for (tag, expected) in canonical {
        assert_eq!(
            parse_cache_type(tag).unwrap(),
            expected,
            "round-trip failed for '{tag}'"
        );
    }
}

#[test]
fn parse_cache_type_alias_f16_resolves_to_bf16() {
    assert_eq!(parse_cache_type("f16").unwrap(), CacheType::Bf16);
}

#[test]
fn parse_cache_type_unknown_tag_returns_error_containing_unknown() {
    let err = parse_cache_type("notacodec").unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("notacodec"),
        "error should name the unknown tag: {msg}"
    );
}

// ── build_cache_type_spec ────────────────────────────────────────────────

#[test]
fn build_cache_type_spec_both_none_returns_none() {
    assert_eq!(build_cache_type_spec(None, None).unwrap(), None);
}

#[test]
fn build_cache_type_spec_k_only_sets_v_auto() {
    let spec = build_cache_type_spec(Some("q8_g128"), None)
        .unwrap()
        .unwrap();
    assert_eq!(
        spec,
        CacheTypeSpec {
            k: CacheType::Q8G128,
            v: CacheType::Auto,
        }
    );
}

#[test]
fn build_cache_type_spec_v_only_sets_k_auto() {
    let spec = build_cache_type_spec(None, Some("tq4")).unwrap().unwrap();
    assert_eq!(
        spec,
        CacheTypeSpec {
            k: CacheType::Auto,
            v: CacheType::Tq4,
        }
    );
}

#[test]
fn build_cache_type_spec_both_some_uses_both() {
    let spec = build_cache_type_spec(Some("q8_g128"), Some("tq4"))
        .unwrap()
        .unwrap();
    assert_eq!(
        spec,
        CacheTypeSpec {
            k: CacheType::Q8G128,
            v: CacheType::Tq4,
        }
    );
}

// ── parse_kv_bits_combo ──────────────────────────────────────────

#[test]
fn kv_bits_8_g128_maps_to_k8v8() {
    let kq = parse_kv_bits_combo(8, 128).unwrap();
    assert_eq!(kq, rmlx_kv_quant::KvQuant::K8V8);
}

#[test]
fn kv_bits_4_g64_maps_to_mixed_k8v4() {
    // mlx-lm default: K stays 8-bit/g=64, V=4-bit/g=64.
    let kq = parse_kv_bits_combo(4, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn kv_bits_4_g32_maps_to_mixed_k8_v4g32() {
    let kq = parse_kv_bits_combo(4, 32).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 32,
        }
    );
}

#[test]
fn kv_bits_8_g64_maps_to_mixed_k8v8_g64() {
    let kq = parse_kv_bits_combo(8, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 8,
            v_bits: 8,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn kv_bits_3_g64_maps_to_mixed() {
    let kq = parse_kv_bits_combo(3, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 8,
            v_bits: 3,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn kv_bits_2_g64_maps_to_mixed_k8_v2() {
    // --kv-bits 2 maps to Mixed{k=8, v=2} (K stays 8-bit; only the V
    // side drops to 2-bit). Pure 2-bit K is gated in combo_to_kv_quant and
    // is unreachable from this integer-alias path.
    let kq = parse_kv_bits_combo(2, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 8,
            v_bits: 2,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn kv_bits_invalid_rejects_7() {
    let err = parse_kv_bits_combo(7, 64).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains('7'),
        "error should name the bad bits value: {msg}"
    );
}

#[test]
fn kv_bits_group_size_zero_rejected() {
    let err = parse_kv_bits_combo(4, 0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("--kv-group-size"),
        "error should mention --kv-group-size: {msg}"
    );
}

// ── parse_kv_bits_fractional ────────────────────────────────────

#[test]
fn fractional_3_5_yields_mixed_k3_v4() {
    // 3.5 → floor=3, ceil=4, both sides use the supplied group_size.
    let kq = parse_kv_bits_fractional(3.5, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 3,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn fractional_4_5_yields_mixed_k4_v5() {
    let kq = parse_kv_bits_fractional(4.5, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 4,
            v_bits: 5,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn fractional_5_5_yields_mixed_k5_v6() {
    let kq = parse_kv_bits_fractional(5.5, 64).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 5,
            v_bits: 6,
            k_group_size: 64,
            v_group_size: 64,
        }
    );
}

#[test]
fn fractional_group_size_propagated_to_both_sides() {
    // Group size propagates to both k and v sides.
    let kq = parse_kv_bits_fractional(3.5, 32).unwrap();
    assert_eq!(
        kq,
        rmlx_kv_quant::KvQuant::Mixed {
            k_bits: 3,
            v_bits: 4,
            k_group_size: 32,
            v_group_size: 32,
        }
    );
}

#[test]
fn fractional_1_5_rejects_k_floor_1() {
    // floor(1.5)=1 is not in {3,4,5,6,8}.
    let err = parse_kv_bits_fractional(1.5, 64).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("K floor=1"),
        "error should name K floor: {msg}"
    );
}

#[test]
fn fractional_7_5_rejects_v_ceil_8_not_actually_err() {
    // floor(7.5)=7 is not in {3,4,5,6,8} → K floor error fires first.
    let err = parse_kv_bits_fractional(7.5, 64).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("K floor=7"),
        "error should name K floor: {msg}"
    );
}

#[test]
fn fractional_group_size_zero_rejected() {
    let err = parse_kv_bits_fractional(3.5, 0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("--kv-group-size"),
        "error should mention --kv-group-size: {msg}"
    );
}

// ── parse_max_prompt_tokens ──────────────────────────────────────

#[test]
fn parse_max_prompt_tokens_rejects_zero() {
    let err = parse_max_prompt_tokens(0).unwrap_err();
    assert!(err.to_string().contains(">= 1"), "{err}");
}

#[test]
fn parse_max_prompt_tokens_accepts_one() {
    assert_eq!(parse_max_prompt_tokens(1).unwrap(), 1);
}

// ── resolve_model_flags fractional dispatch ─────────────────────
// Integer path via resolve_model_flags with f32 must match exactly.

#[test]
fn integer_bits_as_f32_dispatches_to_integer_path() {
    // 4.0 → fract() == 0 → parse_kv_bits_combo(4, 64) → Mixed{k=8,v=4}.
    // We test parse_kv_bits_combo directly for this; the f32 cast is exact.
    let kq_int = parse_kv_bits_combo(4u8, 64).unwrap();
    // Simulate the dispatch in resolve_model_flags for bits=4.0f32:
    let bits: f32 = 4.0;
    let kq_f32 = if bits.fract() == 0.0 {
        parse_kv_bits_combo(bits as u8, 64).unwrap()
    } else {
        parse_kv_bits_fractional(bits, 64).unwrap()
    };
    assert_eq!(kq_int, kq_f32, "f32 integer dispatch must match u8 path");
}

// ── auto KV-quant resolution ─────────────────────────────────────────────
//
// `--kv-quant auto` (i.e. no flag) must resolve to the same codec for every
// architecture the CLI can load. The table below names one checkpoint shape
// per branch the arch resolver used to distinguish, so a per-arch exception
// re-appearing anywhere fails here rather than at a user's first serve.

/// One synthetic `config.json` per architecture class the resolver sees.
fn auto_resolution_fixtures() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "qwen3-vl-moe",
            r#"{"architectures":["Qwen3VLMoeForConditionalGeneration"]}"#,
        ),
        (
            "qwen3.5-moe",
            r#"{"architectures":["Qwen3_5MoeForConditionalGeneration"]}"#,
        ),
        (
            "qwen3.5-dense-paro",
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"],
                "quantization_config":{"quant_method":"paroquant"}}"#,
        ),
        (
            "qwen3.5-dense",
            r#"{"architectures":["Qwen3_5ForConditionalGeneration"]}"#,
        ),
        (
            "qwen3-dense-2bit",
            r#"{"architectures":["Qwen3ForCausalLM"],
                "quantization":{"group_size":64,"bits":2}}"#,
        ),
        (
            "qwen3-dense-8bit",
            r#"{"architectures":["Qwen3ForCausalLM"],
                "quantization":{"group_size":64,"bits":8}}"#,
        ),
        ("qwen2-dense", r#"{"architectures":["Qwen2ForCausalLM"]}"#),
        ("laguna", r#"{"architectures":["LagunaForCausalLM"]}"#),
        (
            "gemma3",
            r#"{"architectures":["Gemma3ForConditionalGeneration"]}"#,
        ),
        (
            "gemma4-moe",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],
                "text_config":{"hidden_size":2048,"enable_moe_block":true}}"#,
        ),
        (
            "gemma4-small-paro",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],
                "text_config":{"hidden_size":1536},
                "quantization_config":{"quant_method":"paroquant"}}"#,
        ),
        (
            "gemma4-small",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],
                "text_config":{"hidden_size":1536}}"#,
        ),
        (
            "gemma4-dense",
            r#"{"architectures":["Gemma4ForConditionalGeneration"],
                "text_config":{"hidden_size":5376}}"#,
        ),
        (
            "gemma4-unified-12b",
            r#"{"architectures":["Gemma4UnifiedForConditionalGeneration"],
                "text_config":{"hidden_size":3840}}"#,
        ),
        (
            "unknown-arch",
            r#"{"architectures":["NoSuchArchForCausalLM"]}"#,
        ),
        ("no-arch-field", r"{}"),
    ]
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "fixture JSON is a literal in this file; a parse failure is a test bug"
)]
fn auto_kv_quant_resolves_to_bf16_for_every_arch_branch() {
    for (name, body) in auto_resolution_fixtures() {
        let cfg: rmlx_loader::ModelConfig = serde_json::from_str(body).unwrap();
        let resolved = resolve_kv_quant(&cfg, None, None);
        assert_eq!(
            resolved,
            rmlx_kv_quant::KvQuant::None,
            "auto resolution for '{name}' is {resolved:?}, not bf16"
        );
    }
}

// ── --paged-kv against the resolved codec ────────────────────────────────────
//
// Paged KV pages a codec's packed store. Under the bf16 auto default there is
// no store, so the flag combination is refused rather than silently promoted to
// a quantised codec — see `reject_paged_kv_without_store`.

#[test]
fn paged_kv_is_refused_when_the_resolved_codec_keeps_no_store() {
    use rmlx_kv_quant::KvQuant;
    // The auto default, and the two ways an operator reaches it.
    for resolved in [None, Some(rmlx_models::kv_cache::DEFAULT_KV_QUANT)] {
        let msg = reject_paged_kv_without_store(true, resolved)
            .unwrap_or_else(|| panic!("--paged-kv with {resolved:?} must be refused"));
        // The message has to be actionable for someone who passed no codec flag.
        assert!(
            msg.contains("--kv-quant auto"),
            "message must explain that auto is bf16: {msg}"
        );
        assert!(
            msg.contains("k8v8"),
            "message must name a codec to pass: {msg}"
        );
    }
    // The bar is "unquantised", unchanged from before the auto default moved:
    // a named quantised codec is still accepted, even though the mirror family
    // no longer materialises a store. Widening that is a separate change.
    assert_eq!(
        reject_paged_kv_without_store(true, Some(KvQuant::K8V8)),
        None
    );
}

#[test]
fn paged_kv_is_accepted_when_a_quantised_codec_was_named() {
    use rmlx_kv_quant::KvQuant;
    let mixed = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    assert!(
        mixed.materialises_packed_store(),
        "fixture must keep a store"
    );
    for kq in [mixed, KvQuant::K8V4, KvQuant::Planar] {
        assert_eq!(
            reject_paged_kv_without_store(true, Some(kq)),
            None,
            "--paged-kv refused a named quantised codec {kq:?}"
        );
    }
}

#[test]
fn without_paged_kv_the_codec_is_never_second_guessed() {
    use rmlx_kv_quant::KvQuant;
    for resolved in [None, Some(KvQuant::None), Some(KvQuant::K8V8)] {
        assert_eq!(
            reject_paged_kv_without_store(false, resolved),
            None,
            "the guard fired without --paged-kv for {resolved:?}"
        );
    }
}

// ── --kv-bits / --kv-group-size are the same set --kv-quant is ───────────

#[test]
fn kv_bits_combo_rejects_a_group_size_the_codec_cannot_store() {
    // `--kv-bits 4 --kv-group-size 17` used to build Mixed{k8g64_v4g17}: a
    // shape `KvQuant::from_str` rejects, so the flag pair could construct a
    // codec the codec's own parser refuses to spell.
    let err = parse_kv_bits_combo(4, 17).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("v_group_size=17"),
        "error should name the rejected group size: {msg}"
    );
}

#[test]
fn kv_bits_fractional_rejects_a_group_size_the_codec_cannot_store() {
    let err = parse_kv_bits_fractional(3.5, 17).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("group_size=17"),
        "error should name the rejected group size: {msg}"
    );
}

#[test]
fn kv_bits_outside_the_codec_field_is_rejected_not_saturated() {
    // `bits as u8` saturates, so 4096 would have arrived as 255 and been
    // rejected for the right reason by accident. The screen makes the
    // rejection the code's, not the cast rule's.
    for bits in [-1.0f32, 256.0, 4096.0, f32::NAN, f32::INFINITY] {
        assert!(
            kv_bits_u8(bits).is_err(),
            "--kv-bits {bits} does not fit the codec's u8 bit-width field"
        );
    }
    for (bits, want) in [(0.0f32, 0u8), (4.0, 4), (255.0, 255)] {
        assert_eq!(kv_bits_u8(bits).ok(), Some(want));
    }
}

#[test]
fn kv_bits_combo_rejects_a_group_size_that_does_not_fit_the_codec_field() {
    // The codec field is a u16. `65664 as u16` is 128, which is a group size
    // the validator accepts and `Display` spells back cleanly — so the wrap
    // produced a working codec at a size nobody asked for, and every check
    // downstream of the cast agreed with it. 65664 and 65600 are the two
    // wraps that land on an accepted size (128 and 64); 65536 lands on 0.
    for group_size in [65536usize, 65600, 65664] {
        let err = parse_kv_bits_combo(8, group_size).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("65535") && msg.contains(&group_size.to_string()),
            "--kv-group-size {group_size} should be rejected by value, not \
             wrapped into a u16: {msg}"
        );
    }
}

#[test]
fn kv_bits_fractional_rejects_a_group_size_that_does_not_fit_the_codec_field() {
    for group_size in [65536usize, 65600, 65664] {
        let err = parse_kv_bits_fractional(3.5, group_size).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("65535") && msg.contains(&group_size.to_string()),
            "--kv-group-size {group_size} should be rejected by value, not \
             wrapped into a u16: {msg}"
        );
    }
}

#[test]
fn kv_bits_combo_never_builds_a_codec_from_str_rejects() {
    // The invariant behind the two rejections above, swept rather than
    // sampled: whatever the alias flags accept must be spellable back through
    // the parser that owns the codec set, or the two disagree about which
    // codecs exist.
    for bits in 0u8..=16 {
        for group_size in [
            0usize, 1, 17, 31, 32, 63, 64, 100, 128, 256, 65536, 65600, 65664,
        ] {
            let Ok(kq) = parse_kv_bits_combo(bits, group_size) else {
                continue;
            };
            let spelled = kq.to_string();
            assert_eq!(
                rmlx_kv_quant::KvQuant::from_str(&spelled).ok(),
                Some(kq),
                "--kv-bits {bits} --kv-group-size {group_size} built {spelled}, \
                 which --kv-quant does not accept"
            );
        }
    }
}

#[test]
fn kv_bits_fractional_never_builds_a_codec_from_str_rejects() {
    for half in 0u8..=16 {
        let bits = f32::from(half) + 0.5;
        for group_size in [
            0usize, 1, 17, 31, 32, 63, 64, 100, 128, 256, 65536, 65600, 65664,
        ] {
            let Ok(kq) = parse_kv_bits_fractional(bits, group_size) else {
                continue;
            };
            let spelled = kq.to_string();
            assert_eq!(
                rmlx_kv_quant::KvQuant::from_str(&spelled).ok(),
                Some(kq),
                "--kv-bits {bits} --kv-group-size {group_size} built {spelled}, \
                 which --kv-quant does not accept"
            );
        }
    }
}
