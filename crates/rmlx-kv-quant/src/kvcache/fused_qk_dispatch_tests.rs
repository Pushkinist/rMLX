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
fn fused_qk_layout_absent_for_codecs_that_keep_no_bf16_k() {
    // The shadow is seeded by re-encoding `decode_fp16_k`. These eight codecs
    // keep no bf16 K at decode (each runs its own flash-decode kernel over the
    // packed ring), so `exit_prefill` never materialises that seed and the
    // fused-QK path can never serve them. A layout for them would describe a
    // buffer nothing allocates.
    for codec in [
        KvQuant::Iso3Sym,
        KvQuant::IsoKOnly3,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly4,
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
    ] {
        assert!(
            !codec.feeds_bf16_k_at_decode(false),
            "{codec:?}: this test's premise is that the codec keeps no bf16 K"
        );
        assert!(
            FusedQkLayout::for_codec(codec, 128)
                .expect("layout result")
                .is_none(),
            "{codec:?}: must have no fused-QK layout — it can never reach that path"
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
        28,
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
    // 43 groups of three 3-bit codes: 387 bits, 13 words per row.
    assert_eq!(l.codes_per_token, 13);
    assert_eq!(l.scales_per_token, 43);
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
    // Same 129 codes at 4 bits: 516 bits, 17 words per row.
    assert_eq!(l.codes_per_token, 17);
    assert_eq!(l.scales_per_token, 43);
    assert!(l.has_norm);
    assert!(l.has_rotor_table);
}

#[test]
fn fused_qk_layout_coverage_matches_the_kernel_table() {
    // Third of the three lists that must agree — `lookup_fused_qk_kernel`,
    // `codec_has_gpu_encoder` and `FusedQkLayout::for_codec`. The first pair
    // is pinned above; this pins the layout against the kernel table.
    //
    // A codec with a kernel but no layout does NOT fall back:
    // `FusedQkShadow::allocate` turns the missing layout into a hard `Err` at
    // the first decode dispatch, so the codec crashes rather than taking the
    // legacy path. Asserted as a biconditional over every variant, not over a
    // list of the codecs already known to agree.
    for &codec in ALL_KV_QUANTS {
        let has_layout = FusedQkLayout::for_codec(codec, 128)
            .unwrap_or_else(|e| panic!("{codec:?}: layout query errored at head_dim=128: {e}"))
            .is_some();
        assert_eq!(
            lookup_fused_qk_kernel(codec).is_some(),
            has_layout,
            "{codec:?}: kernel table and layout table disagree — a codec with a kernel but \
             no layout hard-errors at first dispatch instead of falling back"
        );
    }
}

#[test]
fn fused_qk_table_matches_the_bf16_k_mirror_contract() {
    // `try_fused_qk_dispatch` seeds the head-major shadow by re-encoding
    // `decode_fp16_k`, and `exit_prefill` only materialises that seed when
    // `feeds_bf16_k_at_decode` is true. A codec listed here without the
    // mirror is unreachable at every shape on every arch — the state this
    // table was in for `Iso{3,4}Sym`, `IsoKOnly{3,4}`, `Rotor{3,4}Sym` and
    // `RotorKOnly{3,4}`, all of which decode through their own
    // flash-decode-over-quant kernel instead.
    //
    // Asserted one-directionally: the mirror is necessary, not sufficient
    // (plenty of codecs keep a bf16 K and have no fused-QK kernel at all).
    for &codec in ALL_KV_QUANTS {
        if lookup_fused_qk_kernel(codec).is_some() {
            assert!(
                codec.feeds_bf16_k_at_decode(false),
                "{codec:?}: mapped to a fused-QK kernel but keeps no bf16 K mirror, so the \
                 shadow can never be seeded — the entry is unreachable"
            );
        }
    }
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
