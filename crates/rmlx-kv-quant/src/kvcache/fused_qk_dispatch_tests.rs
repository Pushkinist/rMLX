// Sibling tests for `fused_qk_dispatch.rs`.
//
// CPU-resident smoke + accessor tests. GPU parity / dispatch-counter
// integration tests live in `tests/` because they need real GPU.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

use super::{codec_has_gpu_encoder, lookup_fused_qk_kernel};
use crate::kvcache::fused_qk_shadow::FusedQkLayout;
use crate::kvcache::fused_qk_total_dispatch_count;
use crate::KvQuant;

#[test]
fn fused_qk_total_dispatch_count_is_accessible() {
    // Smoke: function returns without panic and is callable any time.
    let _n = fused_qk_total_dispatch_count();
}

#[test]
fn fused_qk_layout_q8_head_dim_128() {
    let l = FusedQkLayout::for_codec(KvQuant::K8V4, 128)
        .expect("layout result")
        .expect("q8 has fused-QK entry");
    assert_eq!(l.codes_per_token, 32, "head_dim/4 = 32 u32 codes");
    assert_eq!(l.scales_per_token, 1, "head_dim/Q8_GROUP=128 = 1 scale");
    assert!(!l.has_norm, "q8 has no norm sideband");
    assert!(!l.has_rotor_table, "q8 has no rotor table");
}

#[test]
fn fused_qk_layout_q8_head_dim_256() {
    let l = FusedQkLayout::for_codec(KvQuant::K8V8, 256)
        .expect("layout result")
        .expect("q8 has fused-QK entry");
    assert_eq!(l.codes_per_token, 64);
    assert_eq!(l.scales_per_token, 2);
    assert!(!l.has_norm);
    assert!(!l.has_rotor_table);
}

#[test]
fn fused_qk_layout_turbo_sym3_head_dim_128() {
    let l = FusedQkLayout::for_codec(KvQuant::TurboSym3, 128)
        .expect("layout result")
        .expect("turbo3 has fused-QK entry");
    // codes = head_dim * 3 / 32 = 12
    assert_eq!(l.codes_per_token, 12);
    // scales = head_dim / 32 = 4
    assert_eq!(l.scales_per_token, 4);
    assert!(!l.has_norm);
    assert!(!l.has_rotor_table);
}

#[test]
fn fused_qk_layout_turbo_sym4_head_dim_128() {
    let l = FusedQkLayout::for_codec(KvQuant::TurboSym4, 128)
        .expect("layout result")
        .expect("turbo4 has fused-QK entry");
    // codes = head_dim / 8 = 16
    assert_eq!(l.codes_per_token, 16);
    // scales = head_dim / 32 = 4
    assert_eq!(l.scales_per_token, 4);
    assert!(!l.has_norm);
    assert!(!l.has_rotor_table);
}

#[test]
fn fused_qk_layout_iso_codecs_head_dim_128() {
    // All four iso variants share one K-side layout: one u32 word and one f32
    // scale per quaternion block (head_dim / 4 = 32 at D=128), plus the
    // per-token L2 norm sideband. No rotor table — iso's rotation is a single
    // fixed quaternion baked into the kernel header, so `n_groups` (which sizes
    // that table) stays 0.
    for codec in [
        KvQuant::Iso3Sym,
        KvQuant::IsoKOnly3,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly4,
    ] {
        let l = FusedQkLayout::for_codec(codec, 128)
            .expect("layout result")
            .unwrap_or_else(|| panic!("{codec:?} has a fused-QK entry"));
        assert_eq!(l.codes_per_token, 32, "{codec:?}: codes = head_dim / 4");
        assert_eq!(l.scales_per_token, 32, "{codec:?}: scales = head_dim / 4");
        assert!(l.has_norm, "{codec:?}: iso carries a per-token L2 norm");
        assert!(
            !l.has_rotor_table,
            "{codec:?}: iso has no rotor table — its quaternion is fixed"
        );
        assert_eq!(
            l.n_groups, 0,
            "{codec:?}: n_groups sizes the rotor table only"
        );
    }
}

#[test]
fn fused_qk_layout_iso_rejects_head_dim_off_the_quaternion_block() {
    // A head_dim that is not a multiple of 4 would drop a partial trailing
    // quaternion block. That must error rather than silently round down.
    for codec in [KvQuant::Iso3Sym, KvQuant::IsoKOnly4] {
        assert!(
            FusedQkLayout::for_codec(codec, 130).is_err(),
            "{codec:?}: head_dim=130 is not a whole number of quaternion blocks"
        );
    }
}

/// Every [`KvQuant`] variant, with representative parameters for the
/// parameterised ones.
///
/// Pinned to the enum by [`assert_kv_quant_exhaustive`] below — a new variant
/// fails to compile until it is added here, which is what makes the coverage
/// test below able to catch an **omission**. A hand-maintained list on its own
/// cannot: the drift it exists to catch is a codec nobody remembered to list.
const ALL_KV_QUANTS: &[KvQuant] = &[
    KvQuant::K8V4,
    KvQuant::K8V8,
    KvQuant::Planar,
    KvQuant::None,
    KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    },
    KvQuant::RotK {
        v_bits: 4,
        v_group_size: 64,
    },
    KvQuant::RotKTq4V,
    KvQuant::K8VTurbo3,
    KvQuant::TurboSym3,
    KvQuant::TurboSym4,
    KvQuant::Planar3,
    KvQuant::PlanarK,
    KvQuant::K8VTurbo2,
    KvQuant::Iso3,
    KvQuant::Iso4,
    KvQuant::Iso3Sym,
    KvQuant::Iso4Sym,
    KvQuant::IsoKOnly3,
    KvQuant::IsoKOnly4,
    KvQuant::Rotor3,
    KvQuant::Rotor4,
    KvQuant::K8VTurbo3Tcq,
    KvQuant::Rotor3Sym,
    KvQuant::Rotor4Sym,
    KvQuant::RotorKOnly3,
    KvQuant::RotorKOnly4,
    KvQuant::RotorK3Asym {
        v_bits: 4,
        v_group_size: 64,
    },
    KvQuant::RotorK4Asym {
        v_bits: 4,
        v_group_size: 64,
    },
    KvQuant::K8VTurbo2Tcq,
];

/// Compile-time pin: this match is exhaustive, so adding a [`KvQuant`] variant
/// breaks the build here until the author also adds it to [`ALL_KV_QUANTS`].
/// Never called — the type check is the whole point.
#[allow(dead_code, reason = "compile-time exhaustiveness pin, never invoked")]
fn assert_kv_quant_exhaustive(q: KvQuant) {
    match q {
        KvQuant::K8V4
        | KvQuant::K8V8
        | KvQuant::Planar
        | KvQuant::None
        | KvQuant::Mixed { .. }
        | KvQuant::RotK { .. }
        | KvQuant::RotKTq4V
        | KvQuant::K8VTurbo3
        | KvQuant::TurboSym3
        | KvQuant::TurboSym4
        | KvQuant::Planar3
        | KvQuant::PlanarK
        | KvQuant::K8VTurbo2
        | KvQuant::Iso3
        | KvQuant::Iso4
        | KvQuant::Iso3Sym
        | KvQuant::Iso4Sym
        | KvQuant::IsoKOnly3
        | KvQuant::IsoKOnly4
        | KvQuant::Rotor3
        | KvQuant::Rotor4
        | KvQuant::K8VTurbo3Tcq
        | KvQuant::Rotor3Sym
        | KvQuant::Rotor4Sym
        | KvQuant::RotorKOnly3
        | KvQuant::RotorKOnly4
        | KvQuant::RotorK3Asym { .. }
        | KvQuant::RotorK4Asym { .. }
        | KvQuant::K8VTurbo2Tcq => (),
    }
}

#[test]
fn all_kv_quants_list_covers_every_variant() {
    // Guards the guard: the exhaustive match above forces an author to touch
    // this file, but only a count check notices if they touch the match and
    // forget the list.
    assert_eq!(
        ALL_KV_QUANTS.len(),
        29,
        "ALL_KV_QUANTS must list every KvQuant variant — update it (and this count) \
         when a variant is added or removed"
    );
}

#[test]
fn fused_qk_encoder_coverage_matches_the_kernel_table() {
    // `codec_has_gpu_encoder` and `lookup_fused_qk_kernel` are the same fact
    // stated twice. A codec with a kernel but no encoder silently sits on the
    // legacy path; one with an encoder but no kernel errors at dispatch.
    //
    // Asserted as a **biconditional over every variant**, not over a list of
    // the codecs already known to agree: the failure this exists to catch is a
    // codec present in one table and absent from the other, and a hand-picked
    // list would simply not mention it. (Iso was in exactly that state before
    // the flash-decode kernel landed.)
    for &codec in ALL_KV_QUANTS {
        assert_eq!(
            codec_has_gpu_encoder(codec),
            lookup_fused_qk_kernel(codec).is_some(),
            "{codec:?}: encoder coverage and kernel table disagree — a codec must be in \
             both or neither"
        );
    }
}

#[test]
fn fused_qk_layout_rotor3_sym_head_dim_128() {
    // Rotor3 variants produce a layout with norms + rotor table.
    let l = FusedQkLayout::for_codec(KvQuant::Rotor3Sym, 128)
        .expect("layout result")
        .expect("rotor3 has fused-QK entry");
    // n_groups = ceil(128/3) = 43.
    assert_eq!(l.codes_per_token, 43);
    assert_eq!(l.scales_per_token, 43);
    assert_eq!(l.n_groups, 43);
    assert!(l.has_norm, "rotor has per-token norm sideband");
    assert!(l.has_rotor_table, "rotor has static rotor table sideband");
}

#[test]
fn fused_qk_layout_rotor4_sym_head_dim_128() {
    let l = FusedQkLayout::for_codec(KvQuant::Rotor4Sym, 128)
        .expect("layout result")
        .expect("rotor4 has fused-QK entry");
    assert_eq!(l.codes_per_token, 43);
    assert_eq!(l.scales_per_token, 43);
    assert_eq!(l.n_groups, 43);
    assert!(l.has_norm);
    assert!(l.has_rotor_table);
}

#[test]
fn fused_qk_layout_rotor_k_only_3() {
    let l = FusedQkLayout::for_codec(KvQuant::RotorKOnly3, 128)
        .expect("layout result")
        .expect("rotor_k_only_3 has fused-QK entry");
    assert_eq!(l.codes_per_token, 43);
    assert!(l.has_norm);
    assert!(l.has_rotor_table);
}

#[test]
fn fused_qk_layout_rotor_k_only_4() {
    let l = FusedQkLayout::for_codec(KvQuant::RotorKOnly4, 128)
        .expect("layout result")
        .expect("rotor_k_only_4 has fused-QK entry");
    assert_eq!(l.codes_per_token, 43);
    assert!(l.has_norm);
    assert!(l.has_rotor_table);
}

#[test]
fn fused_qk_layout_rotor_k_asym_3() {
    let l = FusedQkLayout::for_codec(
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        128,
    )
    .expect("layout result")
    .expect("rotor_k_asym_3 has fused-QK entry");
    assert_eq!(l.codes_per_token, 43);
    assert!(l.has_norm);
    assert!(l.has_rotor_table);
}

#[test]
fn fused_qk_layout_rotor_k_asym_4() {
    let l = FusedQkLayout::for_codec(
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        128,
    )
    .expect("layout result")
    .expect("rotor_k_asym_4 has fused-QK entry");
    assert_eq!(l.codes_per_token, 43);
    assert!(l.has_norm);
    assert!(l.has_rotor_table);
}

#[test]
fn fused_qk_layout_returns_none_for_non_table_codecs() {
    let r = FusedQkLayout::for_codec(KvQuant::None, 128).expect("Ok result");
    assert!(
        r.is_none(),
        "KvQuant::None has no fused-QK entry; layout must be None"
    );
}

#[test]
fn fused_qk_layout_rejects_head_dim_not_multiple_of_group() {
    // q8 needs head_dim % 128 == 0.  Head_dim=64 must error.
    let err = FusedQkLayout::for_codec(KvQuant::K8V4, 64).expect_err("must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("Q8_GROUP_SIZE") || msg.contains("multiple"),
        "expected group-size error, got: {msg}"
    );
}
