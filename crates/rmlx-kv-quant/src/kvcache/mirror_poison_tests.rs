//! What one decode step reads, observed on the CPU for every codec.
//!
//! The predicates `feeds_bf16_{k,v}_at_decode` and `decode_reads_packed_store`
//! state which buffer a decode step reads. This probe does not read them. It
//! prefills a cache, poisons one buffer, runs one decode step and compares the
//! output with the output of an unpoisoned twin:
//!
//! * the bf16 mirror of one axis, filled with a sentinel;
//! * the packed store of one axis, swapped for a store of the same codec that
//!   holds other rows.
//!
//! A poison that moves the output is a buffer that decode read. The expected
//! answer comes from the literal row in `codec_facts_table.rs`, never from the
//! predicate, so a predicate and an engine path that change together still
//! turn a cell red here.
//!
//! Per axis and per `shares_kv`:
//!
//! * mirror poison moves the output exactly when the row's `feeds_bf16_*`
//!   holds;
//! * store poison moves the output exactly when the row's
//!   `decode_reads_packed_store` holds and the row names a store on that axis.
//!
//! The store poison puts a store in place even where `exit_prefill` built none,
//! so a codec that decodes off its mirror is shown to ignore a store that is
//! there. The mirror poison needs a mirror: a codec with none on an axis has no
//! mirror to ignore, and that cell is decided by what `exit_prefill` built.
//!
//! The decode entry is the one a model layer calls: `update_and_sdpa`, and
//! `update_and_sdpa_shared_source` for a cache that shares its K/V. On the CPU
//! no fused Metal arm is eligible, so the output is what the CPU path reads.
//! The GPU fused arms read the packed store too, but this probe cannot see
//! them; `make gpu-test` is their gate.

use super::core::KvCache;
use super::SharedKv;
use crate::quant::codec_facts_tests::facts_for;
use crate::storage::KvStorage;
use crate::test_utils::{array_bytes, env_lock, f32_arr, fnv1a64, lcg_data, TEST_SEED};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;

const KV_H: i32 = 1;
const HEAD_DIM: i32 = 128;
const MAX_SEQ: i32 = 512;
const PREFILL_SEQ: i32 = 24;
const SCALE: f32 = 0.125;
/// The value a poisoned mirror holds at every position.
const SENTINEL: f32 = 3.0;
/// Seed of the rows a poisoned store holds.
const DONOR_SEED: u64 = 0xd0_0e;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Axis {
    K,
    V,
}

#[derive(Debug, Clone, Copy)]
enum Poison {
    Mirror(Axis),
    Store(Axis),
}

#[allow(
    clippy::expect_used,
    reason = "test driver: every step here is on a shape the codec accepts, so a failure is the defect under test and the panic names it"
)]
fn prefilled(quant: KvQuant, shares_kv: bool) -> KvCache {
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ).with_shares_kv(shares_kv);
    cache.enter_prefill();
    let shape = [1_i32, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n, TEST_SEED), &shape);
    let v = f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &shape);
    cache.update(&k, &v, Device::Cpu).expect("prefill chunk");
    cache.exit_prefill(Device::Cpu).expect("exit_prefill");
    cache
}

/// Fill the mirror of `axis` with [`SENTINEL`]. `false` when there is no
/// mirror on that axis.
#[allow(
    clippy::expect_used,
    reason = "test driver: a cast of a fixture array always evaluates"
)]
fn poison_mirror(cache: &mut KvCache, axis: Axis) -> bool {
    let slot = match axis {
        Axis::K => &mut cache.decode_fp16_k,
        Axis::V => &mut cache.decode_fp16_v,
    };
    let Some(mirror) = slot.as_ref() else {
        return false;
    };
    let shape = mirror.shape();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let sentinel = f32_arr(&vec![SENTINEL; n], &shape)
        .astype(mirror.dtype(), Device::Cpu)
        .expect("sentinel cast");
    *slot = Some(sentinel);
    true
}

/// Swap the store of `axis` for the donor's. `false` when the donor has no
/// store on that axis.
fn swap_side<K, V>(
    k: &mut Option<K>,
    v: &mut Option<V>,
    donor_k: &mut Option<K>,
    donor_v: &mut Option<V>,
    axis: Axis,
) -> bool {
    match axis {
        Axis::K => {
            std::mem::swap(k, donor_k);
            k.is_some()
        }
        Axis::V => {
            std::mem::swap(v, donor_v);
            v.is_some()
        }
    }
}

/// Swap the K store of a K-only variant for the donor's. Such a variant has no
/// V store.
fn swap_k_only<K>(k: &mut Option<K>, donor_k: &mut Option<K>, axis: Axis) -> bool {
    if axis != Axis::K {
        return false;
    }
    std::mem::swap(k, donor_k);
    k.is_some()
}

/// Put the donor's store of `axis` in place of the cache's own. The donor holds
/// other rows at the same length, so a decode step that reads the store sees
/// other values. `false` when the donor has no store on that axis.
///
/// No wildcard over the variant set: a new storage variant must say where its
/// stores are before this probe can run.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the last arm is the pair of two different variants; a donor built for the same codec cannot reach it"
)]
fn poison_store(storage: &mut KvStorage, donor: &mut KvStorage, axis: Axis) -> bool {
    use KvStorage as S;
    match (storage, donor) {
        (S::K8V4 { k, v, .. }, S::K8V4 { k: dk, v: dv, .. })
        | (S::K8VTurbo3 { k, v, .. }, S::K8VTurbo3 { k: dk, v: dv, .. })
        | (S::K8VTurbo3Tcq { k, v, .. }, S::K8VTurbo3Tcq { k: dk, v: dv, .. })
        | (S::K8VTurbo2 { k, v, .. }, S::K8VTurbo2 { k: dk, v: dv, .. })
        | (S::K8VTurbo2Tcq { k, v, .. }, S::K8VTurbo2Tcq { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::K8V8 { k, v, .. }, S::K8V8 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::Planar { k, v, .. }, S::Planar { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::TurboSym3 { k, v, .. }, S::TurboSym3 { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::TurboSym4 { k, v, .. }, S::TurboSym4 { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::IsoV3 { k, v, .. }, S::IsoV3 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::IsoV4 { k, v, .. }, S::IsoV4 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::IsoSym3 { k, v, .. }, S::IsoSym3 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::IsoSym4 { k, v, .. }, S::IsoSym4 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::RotorV3 { k, v, .. }, S::RotorV3 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::RotorV4 { k, v, .. }, S::RotorV4 { k: dk, v: dv, .. }) => swap_side(k, v, dk, dv, axis),
        (S::RotorSym3 { k, v, .. }, S::RotorSym3 { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::RotorSym4 { k, v, .. }, S::RotorSym4 { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::RotorKAsym3 { k, v, .. }, S::RotorKAsym3 { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::RotorKAsym4 { k, v, .. }, S::RotorKAsym4 { k: dk, v: dv, .. }) => {
            swap_side(k, v, dk, dv, axis)
        }
        (S::PlanarK { k, .. }, S::PlanarK { k: dk, .. }) => swap_k_only(k, dk, axis),
        (S::IsoKOnly3 { k, .. }, S::IsoKOnly3 { k: dk, .. }) => swap_k_only(k, dk, axis),
        (S::IsoKOnly4 { k, .. }, S::IsoKOnly4 { k: dk, .. }) => swap_k_only(k, dk, axis),
        (S::RotorKOnly3 { k, .. }, S::RotorKOnly3 { k: dk, .. }) => swap_k_only(k, dk, axis),
        (S::RotorKOnly4 { k, .. }, S::RotorKOnly4 { k: dk, .. }) => swap_k_only(k, dk, axis),
        (S::Mixed { state, .. }, S::Mixed { state: donor, .. }) => swap_side(
            &mut state.keys,
            &mut state.values,
            &mut donor.keys,
            &mut donor.values,
            axis,
        ),
        (S::None { .. }, S::None { .. }) => false,
        (S::Paged { .. }, _) | (_, S::Paged { .. }) => panic!(
            "a codec built the paged storage: the paged routing is latched by the CLI and no \
             test in this binary sets it, so the probe has no cell for it"
        ),
        (storage, donor) => panic!(
            "the donor holds a different storage variant: {} against {}",
            super::helpers::storage_variant_name(storage),
            super::helpers::storage_variant_name(donor)
        ),
    }
}

/// A cache of the same codec whose store holds other rows at the prefill
/// length. It is filled on the decode route, the one CPU route on which every
/// codec writes its store, so it has a store where the prefilled cache may
/// have none.
#[allow(
    clippy::expect_used,
    reason = "test driver: every step here is on a shape the codec accepts, so a failure is the defect under test and the panic names it"
)]
fn donor(quant: KvQuant) -> KvStorage {
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);
    let shape = [1_i32, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&lcg_data(n, DONOR_SEED), &shape);
    let v = f32_arr(&lcg_data(n, DONOR_SEED ^ 0x5a5a), &shape);
    if quant.uses_mixed_path() {
        cache
            .update_and_sdpa(&k, &k, &v, SCALE, "", None, Device::Cpu)
            .expect("donor fill");
    } else {
        cache.update(&k, &v, Device::Cpu).expect("donor fill");
    }
    cache.storage
}

/// One decode step through the entry a model layer calls. The digest covers
/// the attention output and, for a cache that shares its K/V, the pair it hands
/// on. An error is an observation too: a decode that cannot run without a
/// buffer read that buffer.
fn decode_digest(cache: &mut KvCache, shares_kv: bool) -> Result<u64, String> {
    let step = [1_i32, KV_H, 1, HEAD_DIM];
    let n: usize = step.iter().map(|&d| d as usize).product();
    let seed = TEST_SEED.wrapping_add(1);
    let q = f32_arr(&lcg_data(n, seed ^ 0x3c3c), &step);
    let k = f32_arr(&lcg_data(n, seed), &step);
    let v = f32_arr(&lcg_data(n, seed ^ 0x5a5a), &step);
    let mut bytes = Vec::new();
    if shares_kv {
        let (out, shared) = cache
            .update_and_sdpa_shared_source(&q, &k, &v, SCALE, "", None, Device::Cpu)
            .map_err(|e| e.to_string())?;
        bytes.extend_from_slice(&array_bytes(&out));
        match shared {
            SharedKv::Bf16(sk, sv) => {
                bytes.extend_from_slice(&array_bytes(&sk));
                bytes.extend_from_slice(&array_bytes(&sv));
            }
            SharedKv::Store { kv_len } => bytes.extend_from_slice(&kv_len.to_le_bytes()),
        }
    } else {
        let out = cache
            .update_and_sdpa(&q, &k, &v, SCALE, "", None, Device::Cpu)
            .map_err(|e| e.to_string())?;
        bytes.extend_from_slice(&array_bytes(&out));
    }
    Ok(fnv1a64(&bytes))
}

/// What the literal row says about one poison: whether the poisoned buffer is
/// there after the poison, and whether it moves the decode output.
///
/// A mirror is there exactly when decode reads it, because `exit_prefill`
/// builds no mirror that decode does not read. A store is there whenever the
/// row names one on that axis, because the donor puts one in place.
fn row_expects(quant: KvQuant, shares_kv: bool, poison: Poison) -> (bool, bool) {
    let spelling = quant.to_string();
    let Some(row) = facts_for(&spelling) else {
        panic!("{spelling}: no row in codec_facts_table.rs")
    };
    let i = usize::from(shares_kv);
    let reads = row.decode_reads_packed_store;
    match poison {
        Poison::Mirror(Axis::K) => (row.feeds_bf16_k[i], row.feeds_bf16_k[i]),
        Poison::Mirror(Axis::V) => (row.feeds_bf16_v[i], row.feeds_bf16_v[i]),
        Poison::Store(Axis::K) => {
            let named = row.side_stores.0.is_some();
            (named, reads && named)
        }
        Poison::Store(Axis::V) => {
            let named = row.side_stores.1.is_some();
            (named, reads && named)
        }
    }
}

const POISONS: [Poison; 4] = [
    Poison::Mirror(Axis::K),
    Poison::Mirror(Axis::V),
    Poison::Store(Axis::K),
    Poison::Store(Axis::V),
];

#[test]
fn decode_reads_exactly_the_buffers_each_row_names() {
    // The rotor K stores read the QJL switch when they are built. The twins
    // must be one codec, so no writer may move it during the sweep.
    let _guard = env_lock();
    for &quant in ALL_KV_QUANTS {
        for shares_kv in [false, true] {
            let cell = format!("{quant} shares_kv={shares_kv}");
            let baseline = decode_digest(&mut prefilled(quant, shares_kv), shares_kv);
            assert!(
                baseline.is_ok(),
                "{cell}: the unpoisoned decode step failed: {baseline:?}"
            );
            for poison in POISONS {
                let (expect_present, expect_moves) = row_expects(quant, shares_kv, poison);
                let mut cache = prefilled(quant, shares_kv);
                let present = match poison {
                    Poison::Mirror(axis) => poison_mirror(&mut cache, axis),
                    Poison::Store(axis) => {
                        poison_store(&mut cache.storage, &mut donor(quant), axis)
                    }
                };
                assert_eq!(
                    present,
                    expect_present,
                    "{cell} {poison:?}: the poisoned buffer is {}there, but the row says it is \
                     {}there",
                    if present { "" } else { "not " },
                    if expect_present { "" } else { "not " },
                );
                let moved = decode_digest(&mut cache, shares_kv) != baseline;
                assert_eq!(
                    moved,
                    expect_moves,
                    "{cell} {poison:?}: the decode output {} under this poison, but the row \
                     says decode {} this buffer",
                    if moved { "moved" } else { "did not move" },
                    if expect_moves {
                        "reads"
                    } else {
                        "does not read"
                    },
                );
            }
        }
    }
}
