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
fn fused_qk_layout_returns_none_for_iso_codecs() {
    // Iso codecs remain held until their K-side GPU encoder ships.
    // The shadow supports the layout (sideband norms) but the dispatch
    // site has no encoder to populate them. `for_codec` returns `Ok(None)`
    // for iso; dispatch falls through to the legacy bf16 SDPA path.
    for codec in [
        KvQuant::Iso3Sym,
        KvQuant::IsoKOnly3,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly4,
    ] {
        let r = FusedQkLayout::for_codec(codec, 128).expect("Ok result");
        assert!(
            r.is_none(),
            "FusedQkLayout::for_codec({codec:?}) must return None — iso encoder followup"
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
