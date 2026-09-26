//! `KvStorage::max_seq` is the one reader of the per-variant `max_seq`, and
//! every gate that reads it through the accessor keeps its variant set.
//!
//! The references below are the per-variant reads as they stood before the
//! accessor, copied verbatim. For every codec, on an empty and on a filled
//! storage, and for the paged storage, the accessor must return the value the
//! old read returned, and each gate must give the same `Some` / `None` (or
//! `Ok` / `Err`) decision as its old copy.

use super::core::KvCache;
use super::deep_clone_digest_tests::fill;
use super::sdpa::{iso_k_max_seq, rotor_k_max_seq};
use super::store_bytes_tests::TEST_MAX_SEQ;
use crate::storage::KvStorage;
use crate::test_utils::env_lock;
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_core::error::{Error, Result};

/// The paged codecs `KvStorage::new` can wrap in `Paged`.
const PAGED_QUANTS: [KvQuant; 4] = [
    KvQuant::K8V4,
    KvQuant::K8V8,
    KvQuant::Planar,
    KvQuant::Planar3,
];

/// Capacities the empty storages are built with. `0` reaches the `m <= 0`
/// branch of the fused-QK gate.
const EMPTY_MAX_SEQS: [i32; 3] = [0, 1, TEST_MAX_SEQ];

fn old_storage_max_seq(storage: &KvStorage) -> i32 {
    match storage {
        KvStorage::K8V4 { max_seq, .. } => *max_seq,
        KvStorage::K8V8 { max_seq, .. } => *max_seq,
        KvStorage::Planar { max_seq, .. } => *max_seq,
        KvStorage::None { max_seq } => *max_seq,
        KvStorage::Mixed { max_seq, .. } => *max_seq,
        KvStorage::Paged { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo3 { max_seq, .. } => *max_seq,
        KvStorage::TurboSym3 { max_seq, .. } => *max_seq,
        KvStorage::TurboSym4 { max_seq, .. } => *max_seq,
        KvStorage::PlanarK { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo2 { max_seq, .. } => *max_seq,
        KvStorage::IsoV3 { max_seq, .. } => *max_seq,
        KvStorage::IsoV4 { max_seq, .. } => *max_seq,
        KvStorage::RotorV3 { max_seq, .. } => *max_seq,
        KvStorage::RotorV4 { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo3Tcq { max_seq, .. } => *max_seq,
        KvStorage::K8VTurbo2Tcq { max_seq, .. } => *max_seq,
        KvStorage::IsoSym3 { max_seq, .. } => *max_seq,
        KvStorage::IsoSym4 { max_seq, .. } => *max_seq,
        KvStorage::IsoKOnly3 { max_seq, .. } => *max_seq,
        KvStorage::IsoKOnly4 { max_seq, .. } => *max_seq,
        KvStorage::RotorSym3 { max_seq, .. } => *max_seq,
        KvStorage::RotorSym4 { max_seq, .. } => *max_seq,
        KvStorage::RotorKOnly3 { max_seq, .. } => *max_seq,
        KvStorage::RotorKOnly4 { max_seq, .. } => *max_seq,
        KvStorage::RotorKAsym3 { max_seq, .. } => *max_seq,
        KvStorage::RotorKAsym4 { max_seq, .. } => *max_seq,
    }
}

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "a verbatim copy of the fused-QK gate, whose catch-all arm is the variant set under test"
)]
fn old_fused_qk_gate(storage: &KvStorage) -> Option<i32> {
    let m = match storage {
        KvStorage::K8V4 { max_seq, .. } => *max_seq,
        KvStorage::K8V8 { max_seq, .. } => *max_seq,
        KvStorage::TurboSym3 { max_seq, .. } => *max_seq,
        KvStorage::TurboSym4 { max_seq, .. } => *max_seq,
        KvStorage::RotorKAsym3 { max_seq, .. } => *max_seq,
        KvStorage::RotorKAsym4 { max_seq, .. } => *max_seq,
        _ => return None,
    };
    if m <= 0 {
        None
    } else {
        Some(m)
    }
}

fn old_iso_k_max_seq(storage: &KvStorage) -> Result<i32> {
    if let KvStorage::IsoKOnly3 { max_seq, .. } | KvStorage::IsoKOnly4 { max_seq, .. } = storage {
        Ok(*max_seq)
    } else {
        Err(Error::KvStorageMismatch {
            expected: "IsoKOnly3 | IsoKOnly4",
            got: super::helpers::storage_variant_name(storage),
        })
    }
}

fn old_rotor_k_max_seq(storage: &KvStorage) -> Result<i32> {
    if let KvStorage::RotorKOnly3 { max_seq, .. } | KvStorage::RotorKOnly4 { max_seq, .. } = storage
    {
        Ok(*max_seq)
    } else {
        Err(Error::KvStorageMismatch {
            expected: "RotorKOnly3 | RotorKOnly4",
            got: super::helpers::storage_variant_name(storage),
        })
    }
}

fn old_geometry_only_max_seq(storage: &KvStorage) -> Option<i32> {
    fn empty<T>(k: Option<&T>, max_seq: i32) -> Option<i32> {
        k.is_none().then_some(max_seq)
    }
    match storage {
        KvStorage::None { max_seq } => Some(*max_seq),
        KvStorage::K8V4 { k, max_seq, .. }
        | KvStorage::K8V8 { k, max_seq, .. }
        | KvStorage::Planar { k, max_seq, .. }
        | KvStorage::K8VTurbo3 { k, max_seq, .. }
        | KvStorage::K8VTurbo3Tcq { k, max_seq, .. }
        | KvStorage::K8VTurbo2 { k, max_seq, .. }
        | KvStorage::K8VTurbo2Tcq { k, max_seq, .. }
        | KvStorage::IsoV3 { k, max_seq, .. }
        | KvStorage::IsoV4 { k, max_seq, .. }
        | KvStorage::RotorV3 { k, max_seq, .. }
        | KvStorage::RotorV4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::TurboSym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::TurboSym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::PlanarK { k, max_seq } => empty(k.as_ref(), *max_seq),
        KvStorage::IsoSym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::IsoSym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::IsoKOnly3 { k, max_seq } => empty(k.as_ref(), *max_seq),
        KvStorage::IsoKOnly4 { k, max_seq } => empty(k.as_ref(), *max_seq),
        KvStorage::RotorSym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::RotorSym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::RotorKOnly3 { k, max_seq } => empty(k.as_ref(), *max_seq),
        KvStorage::RotorKOnly4 { k, max_seq } => empty(k.as_ref(), *max_seq),
        KvStorage::RotorKAsym3 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::RotorKAsym4 { k, max_seq, .. } => empty(k.as_ref(), *max_seq),
        KvStorage::Mixed { .. } | KvStorage::Paged { .. } => None,
    }
}

/// A result as a comparable value: the `Ok` value, or the error text.
fn outcome(r: Result<i32>) -> std::result::Result<i32, String> {
    r.map_err(|e| e.to_string())
}

/// Every storage the sweep reads, each in a cache, with a label naming it.
fn caches() -> Vec<(String, KvCache)> {
    let mut out = Vec::new();
    for &quant in ALL_KV_QUANTS {
        for max_seq in EMPTY_MAX_SEQS {
            let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
            cache.storage = KvStorage::new(quant, max_seq);
            out.push((format!("{quant} empty at max_seq {max_seq}"), cache));
        }
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        fill(&mut cache, quant);
        out.push((format!("{quant} filled"), cache));
    }
    for quant in PAGED_QUANTS {
        for max_seq in EMPTY_MAX_SEQS {
            let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
            cache.storage = KvStorage::Paged {
                quant,
                k: None,
                v_k8: None,
                v_planar: None,
                max_seq,
            };
            out.push((format!("paged {quant} at max_seq {max_seq}"), cache));
        }
    }
    out
}

#[test]
fn the_accessor_and_every_gate_read_what_the_old_reads_read() {
    // The rotor K stores read the QJL switch when they are built.
    let _guard = env_lock();
    let caches = caches();
    for (label, cache) in &caches {
        let storage = &cache.storage;
        assert_eq!(
            storage.max_seq(),
            old_storage_max_seq(storage),
            "{label}: the accessor"
        );
        assert_eq!(
            cache.storage_max_seq_for_fused_qk(),
            old_fused_qk_gate(storage),
            "{label}: the fused-QK gate"
        );
        assert_eq!(
            outcome(iso_k_max_seq(storage)),
            outcome(old_iso_k_max_seq(storage)),
            "{label}: the iso K-only read"
        );
        assert_eq!(
            outcome(rotor_k_max_seq(storage)),
            outcome(old_rotor_k_max_seq(storage)),
            "{label}: the rotor K-only read"
        );
        assert_eq!(
            storage.geometry_only_max_seq(),
            old_geometry_only_max_seq(storage),
            "{label}: the geometry-only read"
        );
    }
    // Each gate admits some storage and refuses another, so the comparison
    // above is not over one outcome only.
    let admitted = |gate: &dyn Fn(&KvCache) -> bool| caches.iter().filter(|(_, c)| gate(c)).count();
    for (name, n) in [
        (
            "fused-QK",
            admitted(&|c| c.storage_max_seq_for_fused_qk().is_some()),
        ),
        (
            "iso K-only",
            admitted(&|c| iso_k_max_seq(&c.storage).is_ok()),
        ),
        (
            "rotor K-only",
            admitted(&|c| rotor_k_max_seq(&c.storage).is_ok()),
        ),
        (
            "geometry-only",
            admitted(&|c| c.storage.geometry_only_max_seq().is_some()),
        ),
    ] {
        assert!(
            n > 0 && n < caches.len(),
            "the {name} gate admits {n} of {} storages",
            caches.len()
        );
    }
}
