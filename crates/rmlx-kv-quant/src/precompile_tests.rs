//! Tests for issue #36 load-time MSL precompile + codec classification.
//!
//! These are CPU-only / classification tests — they assert the precompile
//! entrypoint is a no-op for `none` and for CPU-hot-path codecs, and that the
//! `carries_msl` / `cpu_hot_path_reason` classifiers are internally consistent.
//! The actual Metal warm dispatch is exercised by the on-device proof in the
//! issue-#36 report, not here (no Metal context in unit tests).

use super::*;
use rmlx_mlx::Device;

#[test]
fn precompile_is_noop_on_cpu_device() {
    // Device::Cpu → always a no-op, regardless of codec.
    for kq in [
        KvQuant::None,
        KvQuant::K8V4,
        KvQuant::Rotor3,
        KvQuant::Iso3,
        KvQuant::RotKTq4V,
    ] {
        assert!(
            precompile_kv_codec_msl(kq, 256, 1, Device::Cpu).is_ok(),
            "precompile on CPU device must be a no-op Ok for {kq}"
        );
    }
}

#[test]
fn none_carries_no_msl() {
    assert!(!KvQuant::None.carries_msl(), "bf16 KV carries no shader");
    assert!(
        KvQuant::None.cpu_hot_path_reason().is_none(),
        "None is not a CPU-fallback codec — it is a no-op bf16 path"
    );
}

#[test]
fn metal_codecs_carry_msl_and_are_not_cpu_fallback() {
    // Codecs whose hot path is genuinely Metal: carry MSL, not CPU-fallback.
    for kq in [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::RotKTq4V,
        KvQuant::TurboSym4,
    ] {
        assert!(kq.carries_msl(), "{kq} must carry MSL");
        assert!(
            kq.cpu_hot_path_reason().is_none(),
            "{kq} runs on Metal — must NOT be flagged CPU-fallback"
        );
    }
}

#[test]
fn iso_rotor_families_are_cpu_fallback_with_reason() {
    // The issue-#36 target codecs: carry MSL (q8 K-side) yet run their codec
    // encode/dequant on CPU — must report a reason.
    for kq in [
        KvQuant::Iso3,
        KvQuant::Iso4,
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::Rotor3,
        KvQuant::Rotor4,
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
        KvQuant::RotorK3Asym {
            v_bits: 8,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 8,
            v_group_size: 128,
        },
    ] {
        assert!(
            kq.carries_msl(),
            "{kq} K-side is q8_0 MSL — carries_msl must be true"
        );
        assert!(
            kq.cpu_hot_path_reason().is_some(),
            "{kq} must be flagged as a CPU-hot-path codec"
        );
    }
}

#[test]
fn precompile_skips_cpu_hot_path_codecs() {
    // Even on a GPU device the precompile is a documented no-op for CPU-fallback
    // codecs (nothing to warm). We can only assert the Ok contract off-device,
    // but the skip branch is taken before any Metal dispatch, so this is safe to
    // run without a Metal context: the function returns early on
    // `cpu_hot_path_reason().is_some()` for these — but only after the device
    // check. We therefore assert via the classifier, which the function consults.
    for kq in [KvQuant::Rotor3, KvQuant::Iso3, KvQuant::IsoKOnly4] {
        assert!(
            kq.cpu_hot_path_reason().is_some(),
            "{kq} drives the precompile skip branch"
        );
    }
}
