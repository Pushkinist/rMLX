//! The read-only sites that go through `KvStorage::view` give the answers of
//! the per-variant matches they replace.
//!
//! The `old_*` fns below are verbatim copies of those matches as they stood
//! before the view: `KvStorage::resident_bytes`, the payload match of
//! `storage_has_materialised_payload`, the two dequant probes and
//! `storage_variant_name`. For every codec in `ALL_KV_QUANTS`, on an empty and
//! on a filled storage, and for a paged storage with and without pages, each
//! view-based answer must equal its old copy. A view arm that drops a slot
//! changes the byte total of the filled storage.
//!
//! `is_geometry_only` has its own old copy in `max_seq_accessor_tests.rs`.
//!
//! The last test is Metal-only: after `eval_gpu_state` every `gpu_*` buffer
//! a store holds is evaluated.
#![allow(
    clippy::too_many_lines,
    clippy::match_same_arms,
    reason = "verbatim copies of the per-variant matches the view replaced"
)]

use super::core::KvCache;
use super::deep_clone_digest_tests::{fill, fill_on};
use super::store_bytes_tests::TEST_MAX_SEQ;
use crate::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};
use crate::storage::{
    KvStorage, QuantIsoK3, QuantIsoK4, QuantIsoV3, QuantIsoV4, QuantK, QuantKTurbo, QuantKTurbo3,
    QuantKTurbo4, QuantPlanarK, QuantPlanarV, QuantRotorK3, QuantRotorK4, QuantRotorV3,
    QuantRotorV4, QuantV,
};
use crate::test_utils::{env_lock, skip_if_no_gpu_env};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_core::error::Result;
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

fn old_resident_bytes(storage: &KvStorage) -> u64 {
    match storage {
        // ── Unquantised (bf16) ─────────────────────────────────────────────
        // Buffers live on KvCache::decode_fp16_k/v; nothing extra here.
        KvStorage::None {} => 0,

        // ── K8V8 (K = q8_0, V = q8_0; V uses QuantK not QuantV) ─────────
        KvStorage::K8V8 { k, v } => {
            opt_bytes(k.as_ref(), QuantK::byte_size) + opt_bytes(v.as_ref(), QuantK::byte_size)
        }

        // ── K8V4 / K8VTurbo* (K = q8_0, V = TurboQuant) ─────────────────
        KvStorage::K8V4 { k, v }
        | KvStorage::K8VTurbo3 { k, v }
        | KvStorage::K8VTurbo3Tcq { k, v }
        | KvStorage::K8VTurbo2 { k, v }
        | KvStorage::K8VTurbo2Tcq { k, v } => {
            opt_bytes(k.as_ref(), QuantK::byte_size) + opt_bytes(v.as_ref(), QuantV::byte_size)
        }

        // ── Planar (K=q8, V=PlanarQuant) ─────────────────────────────────
        KvStorage::Planar { k, v, bits: _ } => {
            opt_bytes(k.as_ref(), QuantK::byte_size)
                + opt_bytes(v.as_ref(), QuantPlanarV::byte_size)
        }

        // ── PlanarK (K=PlanarQuant, V=bf16 on KvCache) ───────────────────
        KvStorage::PlanarK { k } => opt_bytes(k.as_ref(), QuantPlanarK::byte_size),

        // ── Mixed (MLX mx.quantize 3-tuples, opt. RotK) ───────────────────
        KvStorage::Mixed { state } => state.byte_size(),

        // ── Symmetric Turbo (K=TurboK3/4, V=TurboV) ─────────────────────
        KvStorage::TurboSym3 { k, v } => {
            opt_bytes(k.as_ref(), QuantKTurbo3::byte_size)
                + opt_bytes(v.as_ref(), QuantV::byte_size)
        }
        KvStorage::TurboSym4 { k, v } => {
            opt_bytes(k.as_ref(), QuantKTurbo4::byte_size)
                + opt_bytes(v.as_ref(), QuantV::byte_size)
        }

        // ── IsoQuant V (K=q8, V=Iso3/4) ──────────────────────────────────
        KvStorage::IsoV3 { k, v } => {
            opt_bytes(k.as_ref(), QuantK::byte_size) + opt_bytes(v.as_ref(), QuantIsoV3::byte_size)
        }
        KvStorage::IsoV4 { k, v } => {
            opt_bytes(k.as_ref(), QuantK::byte_size) + opt_bytes(v.as_ref(), QuantIsoV4::byte_size)
        }

        // ── IsoQuant Sym (K=IsoK3/4, V=IsoV3/4) ─────────────────────────
        KvStorage::IsoSym3 { k, v } => {
            opt_bytes(k.as_ref(), QuantIsoK3::byte_size)
                + opt_bytes(v.as_ref(), QuantIsoV3::byte_size)
        }
        KvStorage::IsoSym4 { k, v } => {
            opt_bytes(k.as_ref(), QuantIsoK4::byte_size)
                + opt_bytes(v.as_ref(), QuantIsoV4::byte_size)
        }

        // ── IsoKOnly (K=Iso3/4, V=bf16 on KvCache) ───────────────────────
        KvStorage::IsoKOnly3 { k } => opt_bytes(k.as_ref(), QuantIsoK3::byte_size),
        KvStorage::IsoKOnly4 { k } => opt_bytes(k.as_ref(), QuantIsoK4::byte_size),

        // ── RotorV (K=q8, V=Rotor3/4) ────────────────────────────────────
        KvStorage::RotorV3 { k, v } => {
            opt_bytes(k.as_ref(), QuantK::byte_size)
                + opt_bytes(v.as_ref(), QuantRotorV3::byte_size)
        }
        KvStorage::RotorV4 { k, v } => {
            opt_bytes(k.as_ref(), QuantK::byte_size)
                + opt_bytes(v.as_ref(), QuantRotorV4::byte_size)
        }

        // ── RotorSym (K=RotorK3/4, V=RotorV3/4) ─────────────────────────
        KvStorage::RotorSym3 { k, v } => {
            opt_bytes(k.as_ref(), QuantRotorK3::byte_size)
                + opt_bytes(v.as_ref(), QuantRotorV3::byte_size)
        }
        KvStorage::RotorSym4 { k, v } => {
            opt_bytes(k.as_ref(), QuantRotorK4::byte_size)
                + opt_bytes(v.as_ref(), QuantRotorV4::byte_size)
        }

        // ── RotorKOnly (K=RotorK3/4, V=bf16 on KvCache) ─────────────────
        KvStorage::RotorKOnly3 { k } => opt_bytes(k.as_ref(), QuantRotorK3::byte_size),
        KvStorage::RotorKOnly4 { k } => opt_bytes(k.as_ref(), QuantRotorK4::byte_size),

        // ── RotorKAsym (K=RotorK3/4, V=affine QuantV) ────────────────────
        KvStorage::RotorKAsym3 {
            k,
            v,
            v_bits: _,
            v_group_size: _,
        } => {
            opt_bytes(k.as_ref(), QuantRotorK3::byte_size)
                + opt_bytes(v.as_ref(), QuantV::byte_size)
        }
        KvStorage::RotorKAsym4 {
            k,
            v,
            v_bits: _,
            v_group_size: _,
        } => {
            opt_bytes(k.as_ref(), QuantRotorK4::byte_size)
                + opt_bytes(v.as_ref(), QuantV::byte_size)
        }

        // ── Paged (block-table KV, --paged-kv path) ───────────────────────
        KvStorage::Paged {
            k,
            v_k8,
            v_planar,
            quant: _,
        } => {
            let k_bytes = k.as_ref().map_or(0, PagedKStorage::resident_bytes);
            // v_k8 / v_planar are Box-wrapped; closure used to deref through Box.
            let vk8_bytes = v_k8.as_ref().map_or(0, |s| s.resident_bytes());
            let vp_bytes = v_planar.as_ref().map_or(0, |s| s.resident_bytes());
            k_bytes + vk8_bytes + vp_bytes
        }
    }
}

/// Bytes of an optional store slot; an unpopulated slot (`None`) holds nothing.
fn opt_bytes<T>(slot: Option<&T>, byte_size: impl Fn(&T) -> u64) -> u64 {
    slot.map_or(0, byte_size)
}

fn old_storage_payload(storage: &KvStorage) -> bool {
    match storage {
        KvStorage::K8V4 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::K8V8 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::Planar { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::None { .. } => false,
        KvStorage::Mixed { state, .. } => state.offset > 0,
        KvStorage::Paged {
            k, v_k8, v_planar, ..
        } => k.is_some() || v_k8.is_some() || v_planar.is_some(),
        KvStorage::K8VTurbo3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::TurboSym3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::TurboSym4 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::PlanarK { k, .. } => k.is_some(),
        KvStorage::K8VTurbo2 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::IsoV3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::IsoV4 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::RotorV3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::RotorV4 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::K8VTurbo3Tcq { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::K8VTurbo2Tcq { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::IsoSym3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::IsoSym4 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::IsoKOnly3 { k, .. } => k.is_some(),
        KvStorage::IsoKOnly4 { k, .. } => k.is_some(),
        KvStorage::RotorSym3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::RotorSym4 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::RotorKOnly3 { k, .. } => k.is_some(),
        KvStorage::RotorKOnly4 { k, .. } => k.is_some(),
        // RotorKAsym3 / RotorKAsym4 — either side materialised.
        KvStorage::RotorKAsym3 { k, v, .. } => k.is_some() || v.is_some(),
        KvStorage::RotorKAsym4 { k, v, .. } => k.is_some() || v.is_some(),
    }
}

fn old_probe_k(storage: &KvStorage, device: Device) -> Option<Result<Vec<f32>>> {
    match storage {
        KvStorage::K8V4 { k, .. }
        | KvStorage::K8V8 { k, .. }
        | KvStorage::Planar { k, .. }
        | KvStorage::K8VTurbo3 { k, .. }
        | KvStorage::K8VTurbo3Tcq { k, .. }
        | KvStorage::K8VTurbo2Tcq { k, .. }
        | KvStorage::K8VTurbo2 { k, .. }
        | KvStorage::IsoV3 { k, .. }
        | KvStorage::IsoV4 { k, .. }
        | KvStorage::RotorV3 { k, .. }
        | KvStorage::RotorV4 { k, .. } => {
            let k = k.as_ref()?;
            Some(
                k.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // TurboSym3 — K is the 3-bit turbo store. `QuantKTurbo<3>` and
        // `QuantKTurbo<4>` are distinct types, so the two widths dispatch
        // in separate arms even though the call is the same.
        KvStorage::TurboSym3 { k, .. } => {
            let k = k.as_ref()?;
            Some(
                k.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // TurboSym4 — K is the 4-bit turbo store; see the arm above for
        // why the two widths are not one arm.
        KvStorage::TurboSym4 { k, .. } => {
            let k = k.as_ref()?;
            Some(
                k.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // PlanarK — K is a `QuantPlanarK`. Same API as QuantPlanarV.
        KvStorage::PlanarK { k, .. } => {
            let k = k.as_ref()?;
            Some(
                k.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // Iso symmetric / K-only — K is QuantIsoK3 / QuantIsoK4
        // (CPU-only). Dequant via the codec's `dequant()` method.
        KvStorage::IsoSym3 { k, .. } | KvStorage::IsoKOnly3 { k, .. } => {
            Some(k.as_ref()?.dequant())
        }
        KvStorage::IsoSym4 { k, .. } | KvStorage::IsoKOnly4 { k, .. } => {
            Some(k.as_ref()?.dequant())
        }
        // Rotor symmetric / K-only — K is QuantRotorK3 / QuantRotorK4.
        // RotorKAsym3 / RotorKAsym4 — same K codec types.
        KvStorage::RotorSym3 { k, .. }
        | KvStorage::RotorKOnly3 { k, .. }
        | KvStorage::RotorKAsym3 { k, .. } => Some(k.as_ref()?.dequant()),
        KvStorage::RotorSym4 { k, .. }
        | KvStorage::RotorKOnly4 { k, .. }
        | KvStorage::RotorKAsym4 { k, .. } => Some(k.as_ref()?.dequant()),
        KvStorage::None { .. } | KvStorage::Mixed { .. } | KvStorage::Paged { .. } => None,
    }
}

fn old_probe_v(storage: &KvStorage, device: Device) -> Option<Result<Vec<f32>>> {
    match storage {
        // V is a `QuantV` (TurboQuant, any bit width).
        KvStorage::K8V4 { v, .. }
        | KvStorage::K8VTurbo3 { v, .. }
        | KvStorage::K8VTurbo3Tcq { v, .. }
        | KvStorage::K8VTurbo2 { v, .. }
        | KvStorage::K8VTurbo2Tcq { v, .. }
        | KvStorage::TurboSym3 { v, .. }
        | KvStorage::TurboSym4 { v, .. }
        | KvStorage::RotorKAsym3 { v, .. }
        | KvStorage::RotorKAsym4 { v, .. } => {
            let v = v.as_ref()?;
            Some(
                v.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // V is a second `QuantK` (affine q8_0 on both axes).
        KvStorage::K8V8 { v, .. } => {
            let v = v.as_ref()?;
            Some(
                v.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // V is a `QuantPlanarV`.
        KvStorage::Planar { v, .. } => {
            let v = v.as_ref()?;
            Some(
                v.dequantize_choice(device, Dtype::F32)
                    .map(|(flat, _)| flat),
            )
        }
        // Iso / rotor V stores expose `dequant()` rather than the
        // device-choosing pair.
        KvStorage::IsoV3 { v, .. } | KvStorage::IsoSym3 { v, .. } => Some(v.as_ref()?.dequant()),
        KvStorage::IsoV4 { v, .. } | KvStorage::IsoSym4 { v, .. } => Some(v.as_ref()?.dequant()),
        KvStorage::RotorV3 { v, .. } | KvStorage::RotorSym3 { v, .. } => {
            Some(v.as_ref()?.dequant())
        }
        KvStorage::RotorV4 { v, .. } | KvStorage::RotorSym4 { v, .. } => {
            Some(v.as_ref()?.dequant())
        }
        // V is bf16 on the parent cache, or there is no per-axis store.
        KvStorage::PlanarK { .. }
        | KvStorage::IsoKOnly3 { .. }
        | KvStorage::IsoKOnly4 { .. }
        | KvStorage::RotorKOnly3 { .. }
        | KvStorage::RotorKOnly4 { .. }
        | KvStorage::None { .. }
        | KvStorage::Mixed { .. }
        | KvStorage::Paged { .. } => None,
    }
}

fn old_storage_variant_name(s: &KvStorage) -> &'static str {
    match s {
        KvStorage::K8V4 { .. } => "K8V4",
        KvStorage::K8V8 { .. } => "K8V8",
        KvStorage::Planar { .. } => "Planar",
        KvStorage::None { .. } => "None",
        KvStorage::Mixed { .. } => "Mixed",
        KvStorage::Paged { .. } => "Paged",
        KvStorage::K8VTurbo3 { .. } => "K8VTurbo3",
        KvStorage::TurboSym3 { .. } => "TurboSym3",
        KvStorage::TurboSym4 { .. } => "TurboSym4",
        KvStorage::PlanarK { .. } => "PlanarK",
        KvStorage::K8VTurbo2 { .. } => "K8VTurbo2",
        KvStorage::IsoV3 { .. } => "IsoV3",
        KvStorage::IsoV4 { .. } => "IsoV4",
        KvStorage::RotorV3 { .. } => "RotorV3",
        KvStorage::RotorV4 { .. } => "RotorV4",
        KvStorage::K8VTurbo3Tcq { .. } => "K8VTurbo3Tcq",
        KvStorage::K8VTurbo2Tcq { .. } => "K8VTurbo2Tcq",
        KvStorage::IsoSym3 { .. } => "IsoSym3",
        KvStorage::IsoSym4 { .. } => "IsoSym4",
        KvStorage::IsoKOnly3 { .. } => "IsoKOnly3",
        KvStorage::IsoKOnly4 { .. } => "IsoKOnly4",
        KvStorage::RotorSym3 { .. } => "RotorSym3",
        KvStorage::RotorSym4 { .. } => "RotorSym4",
        KvStorage::RotorKOnly3 { .. } => "RotorKOnly3",
        KvStorage::RotorKOnly4 { .. } => "RotorKOnly4",
        KvStorage::RotorKAsym3 { .. } => "RotorKAsym3",
        KvStorage::RotorKAsym4 { .. } => "RotorKAsym4",
    }
}

const PAGE_TOKENS: i32 = 16;
const PAGE_SLOTS: usize = 4;

fn paged_cache(storage: KvStorage) -> KvCache {
    KvCache::from_storage(
        storage,
        TEST_MAX_SEQ,
        KvQuant::K8V4,
        0,
        0,
        DispatchPolicy::default(),
        false,
    )
}

/// A paged storage whose three slots hold 1, 2 and 3 CPU pages, so each slot
/// adds a different byte count to the total.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a CPU page allocation must succeed, and the panic names the slot"
)]
fn paged_with_pages() -> KvStorage {
    let mut k = PagedKStorage::new(TEST_MAX_SEQ, PAGE_TOKENS, PAGE_SLOTS);
    k.codes.alloc(Device::Cpu).expect("allocate a K page");
    let mut v_k8 = PagedVStorage::new(TEST_MAX_SEQ, PAGE_TOKENS, PAGE_SLOTS, 8);
    for _ in 0..2 {
        v_k8.codes.alloc(Device::Cpu).expect("allocate a V page");
    }
    let mut v_planar = PagedPlanarVStorage::new(TEST_MAX_SEQ, PAGE_TOKENS, PAGE_SLOTS);
    for _ in 0..3 {
        v_planar
            .rotations
            .alloc(Device::Cpu)
            .expect("allocate a planar V page");
    }
    KvStorage::Paged {
        quant: KvQuant::K8V4,
        k: Some(k),
        v_k8: Some(Box::new(v_k8)),
        v_planar: Some(Box::new(v_planar)),
    }
}

type ProbeBits = Option<std::result::Result<Vec<u32>, String>>;

fn probe_bits(probe: Option<Result<Vec<f32>>>) -> ProbeBits {
    probe.map(|result| {
        result
            .map(|flat| flat.iter().map(|x| x.to_bits()).collect())
            .map_err(|err| err.to_string())
    })
}

fn assert_view_answers_as_old(label: &str, cache: &KvCache) {
    let storage = &cache.storage;
    assert_eq!(
        storage.resident_bytes(),
        old_resident_bytes(storage),
        "{label}: resident_bytes"
    );
    let filled = storage
        .view()
        .slots
        .iter()
        .flatten()
        .any(|slot| slot.is_filled());
    assert_eq!(filled, old_storage_payload(storage), "{label}: payload");
    assert_eq!(
        storage.view().name,
        old_storage_variant_name(storage),
        "{label}: variant name"
    );
    assert_eq!(
        probe_bits(cache.probe_k_dequant(Device::Cpu)),
        probe_bits(old_probe_k(storage, Device::Cpu)),
        "{label}: K probe"
    );
    assert_eq!(
        probe_bits(cache.probe_v_dequant(Device::Cpu)),
        probe_bits(old_probe_v(storage, Device::Cpu)),
        "{label}: V probe"
    );
}

#[test]
fn view_answers_as_the_old_matches_for_every_codec() {
    // The rotor K stores read the QJL switch when they are built.
    let _guard = env_lock();
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        assert_view_answers_as_old(&format!("{quant} empty"), &cache);
        fill(&mut cache, quant);
        // `None` keeps its KV on the parent cache and has no store.
        assert!(
            quant == KvQuant::None || cache.storage.resident_bytes() > 0,
            "{quant}: the fill wrote no store, so this cell compares nothing"
        );
        assert_view_answers_as_old(&format!("{quant} filled"), &cache);
    }
    let empty = KvStorage::Paged {
        quant: KvQuant::K8V4,
        k: None,
        v_k8: None,
        v_planar: None,
    };
    assert_view_answers_as_old("paged empty", &paged_cache(empty));
    let paged = paged_cache(paged_with_pages());
    assert_ne!(
        paged.storage.resident_bytes(),
        0,
        "paged: no page was allocated"
    );
    assert_view_answers_as_old("paged with pages", &paged);
}

/// `eval_gpu_state` flushes every slot of every filled codec without error.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: the panic names the codec whose flush failed"
)]
fn eval_gpu_state_flushes_every_filled_codec() {
    let _guard = env_lock();
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        fill(&mut cache, quant);
        cache
            .eval_gpu_state()
            .unwrap_or_else(|err| panic!("{quant}: eval_gpu_state failed: {err}"));
    }
    paged_cache(paged_with_pages())
        .eval_gpu_state()
        .expect("paged: eval_gpu_state");
}

type Buffers<'a> = Vec<(&'static str, &'a Option<Array>)>;

fn affine_k(store: Option<&QuantK>) -> Buffers<'_> {
    store
        .iter()
        .flat_map(|s| {
            [
                ("QuantK codes", &s.gpu_codes_buf),
                ("QuantK scales", &s.gpu_scales_buf),
            ]
        })
        .collect()
}

fn turbo_v(store: Option<&QuantV>) -> Buffers<'_> {
    store
        .iter()
        .flat_map(|s| {
            [
                ("QuantV codes", &s.gpu_codes_buf),
                ("QuantV scales", &s.gpu_scales_buf),
            ]
        })
        .collect()
}

fn turbo_k<const BITS: u8>(store: Option<&QuantKTurbo<BITS>>) -> Buffers<'_> {
    store
        .iter()
        .flat_map(|s| {
            [
                ("QuantKTurbo codes", &s.gpu_codes_buf),
                ("QuantKTurbo scales", &s.gpu_scales_buf),
            ]
        })
        .collect()
}

fn planar_k(store: Option<&QuantPlanarK>) -> Buffers<'_> {
    store
        .iter()
        .flat_map(|s| {
            [
                ("QuantPlanarK codes", &s.gpu_codes_buf),
                ("QuantPlanarK scales", &s.gpu_scales_buf),
                ("QuantPlanarK rotations", &s.gpu_rotations_buf),
            ]
        })
        .collect()
}

fn planar_v(store: Option<&QuantPlanarV>) -> Buffers<'_> {
    store
        .iter()
        .flat_map(|s| {
            [
                ("QuantPlanarV codes", &s.gpu_codes_buf),
                ("QuantPlanarV scales", &s.gpu_scales_buf),
                ("QuantPlanarV rotations", &s.gpu_rotations_buf),
            ]
        })
        .collect()
}

/// Every `gpu_*` buffer of the stores whose `eval` flushes it, read from the
/// store fields and not from the `eval` bodies, so a buffer an `eval` body
/// leaves out is still listed here. The iso and rotor stores, `Mixed` and the
/// paged pages are not listed: their `eval` flushes no `gpu_*` field.
fn gpu_buffers(storage: &KvStorage) -> Buffers<'_> {
    match storage {
        KvStorage::K8V4 { k, v }
        | KvStorage::K8VTurbo3 { k, v }
        | KvStorage::K8VTurbo3Tcq { k, v }
        | KvStorage::K8VTurbo2 { k, v }
        | KvStorage::K8VTurbo2Tcq { k, v } => [affine_k(k.as_ref()), turbo_v(v.as_ref())].concat(),
        KvStorage::K8V8 { k, v } => [affine_k(k.as_ref()), affine_k(v.as_ref())].concat(),
        KvStorage::Planar { k, v, bits: _ } => {
            [affine_k(k.as_ref()), planar_v(v.as_ref())].concat()
        }
        KvStorage::PlanarK { k } => planar_k(k.as_ref()),
        KvStorage::TurboSym3 { k, v } => [turbo_k(k.as_ref()), turbo_v(v.as_ref())].concat(),
        KvStorage::TurboSym4 { k, v } => [turbo_k(k.as_ref()), turbo_v(v.as_ref())].concat(),
        KvStorage::IsoV3 { k, v: _ }
        | KvStorage::IsoV4 { k, v: _ }
        | KvStorage::RotorV3 { k, v: _ }
        | KvStorage::RotorV4 { k, v: _ } => affine_k(k.as_ref()),
        KvStorage::RotorKAsym3 { v, .. } | KvStorage::RotorKAsym4 { v, .. } => turbo_v(v.as_ref()),
        KvStorage::None {}
        | KvStorage::Mixed { .. }
        | KvStorage::Paged { .. }
        | KvStorage::IsoSym3 { .. }
        | KvStorage::IsoSym4 { .. }
        | KvStorage::IsoKOnly3 { .. }
        | KvStorage::IsoKOnly4 { .. }
        | KvStorage::RotorSym3 { .. }
        | KvStorage::RotorSym4 { .. }
        | KvStorage::RotorKOnly3 { .. }
        | KvStorage::RotorKOnly4 { .. } => Vec::new(),
    }
}

/// After `eval_gpu_state`, every `gpu_*` buffer a Metal fill left in a store
/// is materialised. A store `eval` that leaves one buffer out leaves it an
/// unevaluated graph node.
///
/// Metal-only: a CPU fill builds no `gpu_*` buffer, so on the CPU this reads
/// nothing.
#[test]
#[ignore = "GPU Metal context — run via `make gpu-test CRATE=rmlx-kv-quant FILTER=storage_view`"]
#[allow(
    clippy::expect_used,
    reason = "test driver: the panic names the codec and the buffer"
)]
fn eval_gpu_state_materialises_every_gpu_buffer_of_every_codec() {
    if skip_if_no_gpu_env() {
        return;
    }
    let _guard = env_lock();
    let mut checked = 0_usize;
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        fill_on(&mut cache, quant, Device::Gpu);
        cache
            .eval_gpu_state()
            .unwrap_or_else(|err| panic!("{quant}: eval_gpu_state failed: {err}"));
        for (buffer, array) in gpu_buffers(&cache.storage) {
            let Some(array) = array else { continue };
            let available = array.is_available().expect("is_available");
            assert!(
                available,
                "{quant}: {buffer} is not evaluated after eval_gpu_state"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "no Metal fill left a gpu_* buffer, so this test read nothing"
    );
}
