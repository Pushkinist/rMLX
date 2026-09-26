//! Every gate that reads `max_seq` keeps its variant set now that the value
//! lives on `KvCache`.
//!
//! The references below are the gates as they stood when `max_seq` was a
//! field of each `KvStorage` variant. Their variant sets are copied verbatim;
//! the value they return is the capacity the cache holds. For every codec, on
//! an empty and on a filled storage, and for the paged storage, each gate must
//! give the same `Some` / `None` decision and value as its old copy.

use super::core::KvCache;
use super::deep_clone_digest_tests::fill;
use super::store_bytes_tests::TEST_MAX_SEQ;
use crate::storage::KvStorage;
use crate::test_utils::env_lock;
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_core::DispatchPolicy;

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

#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "a verbatim copy of the fused-QK gate, whose catch-all arm is the variant set under test"
)]
fn old_fused_qk_gate(storage: &KvStorage, max_seq: i32) -> Option<i32> {
    let m = match storage {
        KvStorage::K8V4 { .. } => max_seq,
        KvStorage::K8V8 { .. } => max_seq,
        KvStorage::TurboSym3 { .. } => max_seq,
        KvStorage::TurboSym4 { .. } => max_seq,
        KvStorage::RotorKAsym3 { .. } => max_seq,
        KvStorage::RotorKAsym4 { .. } => max_seq,
        _ => return None,
    };
    if m <= 0 {
        None
    } else {
        Some(m)
    }
}

fn old_geometry_only_max_seq(storage: &KvStorage, max_seq: i32) -> Option<i32> {
    fn empty<T>(k: Option<&T>, max_seq: i32) -> Option<i32> {
        k.is_none().then_some(max_seq)
    }
    match storage {
        KvStorage::None { .. } => Some(max_seq),
        KvStorage::K8V4 { k, .. }
        | KvStorage::K8V8 { k, .. }
        | KvStorage::Planar { k, .. }
        | KvStorage::K8VTurbo3 { k, .. }
        | KvStorage::K8VTurbo3Tcq { k, .. }
        | KvStorage::K8VTurbo2 { k, .. }
        | KvStorage::K8VTurbo2Tcq { k, .. }
        | KvStorage::IsoV3 { k, .. }
        | KvStorage::IsoV4 { k, .. }
        | KvStorage::RotorV3 { k, .. }
        | KvStorage::RotorV4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::TurboSym3 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::TurboSym4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::PlanarK { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::IsoSym3 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::IsoSym4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::IsoKOnly3 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::IsoKOnly4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::RotorSym3 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::RotorSym4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::RotorKOnly3 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::RotorKOnly4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::RotorKAsym3 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::RotorKAsym4 { k, .. } => empty(k.as_ref(), max_seq),
        KvStorage::Mixed { .. } | KvStorage::Paged { .. } => None,
    }
}

/// Every storage the sweep reads, each in a cache, with a label naming it.
fn caches() -> Vec<(String, KvCache)> {
    let mut out = Vec::new();
    for &quant in ALL_KV_QUANTS {
        for max_seq in EMPTY_MAX_SEQS {
            let cache = KvCache::with_quant_max_seq(quant, max_seq);
            out.push((format!("{quant} empty at max_seq {max_seq}"), cache));
        }
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        fill(&mut cache, quant);
        out.push((format!("{quant} filled"), cache));
    }
    for quant in PAGED_QUANTS {
        for max_seq in EMPTY_MAX_SEQS {
            let storage = KvStorage::Paged {
                quant,
                k: None,
                v_k8: None,
                v_planar: None,
            };
            let cache = KvCache::from_storage(
                storage,
                max_seq,
                quant,
                0,
                0,
                DispatchPolicy::default(),
                false,
            );
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
        let max_seq = cache.max_seq();
        assert_eq!(
            cache.storage_max_seq_for_fused_qk(),
            old_fused_qk_gate(storage, max_seq),
            "{label}: the fused-QK gate"
        );
        assert_eq!(
            storage.is_geometry_only().then_some(max_seq),
            old_geometry_only_max_seq(storage, max_seq),
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
        ("geometry-only", admitted(&|c| c.storage.is_geometry_only())),
    ] {
        assert!(
            n > 0 && n < caches.len(),
            "the {name} gate admits {n} of {} storages",
            caches.len()
        );
    }
}
