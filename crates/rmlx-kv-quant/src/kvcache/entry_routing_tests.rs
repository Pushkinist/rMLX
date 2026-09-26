//! `update` takes its entry from the storage variant. `exit_prefill` takes its
//! entry from the codec, the same key its `materialises_packed_store` gate
//! reads.
//!
//! Two shapes make the storage and the codec disagree in production: an
//! SSD-hydrated SWA layer holds `KvStorage::None` while its `quant` is the
//! model's codec, and `KvStorage::Paged` comes from a process-global switch.
//! The routing tests drive a cache of that shape and a reference cache whose
//! storage and `quant` agree through the same prefill and decode steps, and
//! compare what they return and hold. An `update` entry keyed on `quant`
//! reaches a body that expects another storage and fails. `exit_prefill`
//! finishes both shapes at its guards, before the entry.

use super::core::KvCache;
use crate::storage::KvStorage;
use crate::test_utils::{array_bytes, env_lock, f32_arr, fnv1a64, lcg_data, TEST_SEED};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::Device;

const MAX_SEQ: i32 = 512;
const PREFILL_SEQ: i32 = 24;
const DECODE_STEPS: u64 = 3;
const KV_H: i32 = 2;
const HEAD_DIM: i32 = 128;

/// What one drive leaves behind.
#[derive(Debug, PartialEq, Eq)]
struct Observed {
    offset: i32,
    /// Digest of the K and V rows every decode step returned.
    rows: u64,
    storage: &'static str,
    store_bytes: u64,
    resident_bytes: u64,
}

/// `enter_prefill`, one chunk, `exit_prefill`, then [`DECODE_STEPS`] decode
/// steps through `update`.
fn drive(mut cache: KvCache) -> Result<Observed> {
    let device = Device::Cpu;
    let chunk = [1_i32, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n: usize = chunk.iter().map(|&d| d as usize).product();
    cache.enter_prefill();
    cache.update(
        &f32_arr(&lcg_data(n, TEST_SEED), &chunk),
        &f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &chunk),
        device,
    )?;
    cache.exit_prefill(device)?;

    let step = [1_i32, KV_H, 1, HEAD_DIM];
    let ns: usize = step.iter().map(|&d| d as usize).product();
    let mut rows = Vec::new();
    for i in 0..DECODE_STEPS {
        let seed = TEST_SEED.wrapping_add(i + 1);
        let (k, v) = cache.update(
            &f32_arr(&lcg_data(ns, seed), &step),
            &f32_arr(&lcg_data(ns, seed ^ 0x5a5a), &step),
            device,
        )?;
        rows.extend_from_slice(&array_bytes(&k));
        rows.extend_from_slice(&array_bytes(&v));
    }
    Ok(Observed {
        offset: cache.offset,
        rows: fnv1a64(&rows),
        storage: cache.storage.view().name,
        store_bytes: cache.storage.resident_bytes(),
        resident_bytes: cache.resident_bytes(),
    })
}

fn hydrated(storage: KvStorage, quant: KvQuant) -> KvCache {
    KvCache::from_storage(
        storage,
        MAX_SEQ,
        quant,
        0,
        0,
        DispatchPolicy::default(),
        false,
    )
}

fn empty_paged(quant: KvQuant) -> KvStorage {
    KvStorage::Paged {
        quant,
        k: None,
        v_k8: None,
        v_planar: None,
    }
}

/// A hydrated `None` storage takes the `None` entries whatever `quant` says.
/// `K8V8` is the shape a hydrated SWA layer has; `Iso3Sym` builds a packed
/// store, so it also passes the `materialises_packed_store` gate and only the
/// `None` guard in `exit_prefill` stops it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: the reference cache is a plain bf16 cache, so a failure is the defect under test"
)]
fn hydrated_none_storage_takes_the_none_entries_whatever_the_quant() {
    let _guard = env_lock();
    let reference = drive(KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ)).expect("bf16 drive");
    assert_eq!(reference.storage, "None");
    assert_eq!(reference.store_bytes, 0);
    for quant in [KvQuant::K8V8, KvQuant::Iso3Sym] {
        let got = drive(hydrated(KvStorage::None {}, quant));
        assert_eq!(
            got.as_ref().ok(),
            Some(&reference),
            "None storage with quant {quant}: {got:?} — an entry keyed on the quant, or a \
             missing None guard in exit_prefill"
        );
    }
}

/// A `Paged` storage takes the paged entries whatever `quant` says.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: the reference cache's storage and quant agree, so a failure is the defect under test"
)]
fn paged_storage_takes_the_paged_entries_whatever_the_quant() {
    let _guard = env_lock();
    let reference =
        drive(hydrated(empty_paged(KvQuant::K8V8), KvQuant::K8V8)).expect("paged drive");
    assert_eq!(reference.storage, "Paged");
    for quant in [
        KvQuant::None,
        KvQuant::K8V4,
        KvQuant::Iso3Sym,
        KvQuant::RotorKOnly3,
    ] {
        let got = drive(hydrated(empty_paged(KvQuant::K8V8), quant));
        assert_eq!(
            got.as_ref().ok(),
            Some(&reference),
            "Paged storage with quant {quant}: {got:?} — an entry keyed on the quant"
        );
    }
}

/// The two entries a caller cannot reach refuse with a typed error rather
/// than run a body for another storage.
#[test]
fn the_entries_behind_a_guard_refuse_a_direct_call() {
    let device = Device::Cpu;
    let a = f32_arr(&[0.0; 4], &[1, 1, 1, 4]);
    for storage in [KvStorage::None {}, empty_paged(KvQuant::K8V8)] {
        let mut cache = hydrated(storage, KvQuant::K8V8);
        let entry = cache.storage.view_mut().exit_prefill;
        let err = entry(&mut cache, &a, &a, device, 1);
        assert!(
            err.is_err(),
            "the {} exit_prefill entry must refuse",
            cache.storage.view().name
        );
    }
    let mut mixed = KvCache::with_quant_max_seq(
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
        MAX_SEQ,
    );
    assert!(mixed.update(&a, &a, device).is_err());
}

/// `exit_prefill` takes its entry from the codec, so a storage of another
/// family is refused as a mismatch. Here a K8V8 storage sits beside `Iso3Sym`,
/// whose gate says to build a store: the `Iso3Sym` entry finds K8V8 storage
/// and returns `KvStorageMismatch`. An entry keyed on the storage would build a
/// K8V8 store instead.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: the prefill chunk is a raw append every storage accepts"
)]
fn exit_prefill_refuses_a_storage_of_another_family_than_the_codec() {
    let _guard = env_lock();
    let device = Device::Cpu;
    let storage = KvStorage::K8V8 { k: None, v: None };
    let mut cache = hydrated(storage, KvQuant::Iso3Sym);
    let chunk = [1_i32, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n: usize = chunk.iter().map(|&d| d as usize).product();
    cache.enter_prefill();
    cache
        .update(
            &f32_arr(&lcg_data(n, TEST_SEED), &chunk),
            &f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &chunk),
            device,
        )
        .expect("prefill chunk");
    let got = cache.exit_prefill(device);
    assert!(
        matches!(
            got,
            Err(Error::KvStorageMismatch {
                expected: "IsoSym3 | IsoSym4",
                got: "K8V8",
            })
        ),
        "exit_prefill on K8V8 storage beside Iso3Sym: {got:?}"
    );
}

/// Every `exit_prefill` entry `view_mut` names builds a store when it is
/// called directly on the storage its codec builds. The gate in
/// `exit_prefill` hides the mirror-family entries, so only a direct call
/// reaches them. `None` storage has no store, and its entry refuses.
#[test]
fn every_exit_prefill_entry_builds_a_store_on_its_own_storage() {
    let _guard = env_lock();
    let device = Device::Cpu;
    let chunk = [1_i32, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n: usize = chunk.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n, TEST_SEED), &chunk);
    let v = f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &chunk);
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);
        let entry = cache.storage.view_mut().exit_prefill;
        let got = entry(&mut cache, &k, &v, device, PREFILL_SEQ);
        if quant == KvQuant::None {
            assert!(got.is_err(), "the None exit_prefill entry must refuse");
            continue;
        }
        assert!(
            got.is_ok(),
            "{quant}: the exit_prefill entry failed: {got:?}"
        );
        assert!(
            !cache.storage.is_geometry_only(),
            "{quant}: the exit_prefill entry built no store"
        );
    }
}
