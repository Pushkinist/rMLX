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
