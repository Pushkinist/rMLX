//! `KvStorage::try_deep_clone` copies every byte of a filled store.
//!
//! Each codec in `ALL_KV_QUANTS` is filled on the decode route that writes a
//! store for every codec, then cloned. The store-bytes oracle's reader
//! (`store_digest`) must give the same digest for the source and the clone. A
//! clone that drops a slot, a block or a scalar gives a different digest.
//!
//! The clone must also be deep: after one more append to the source, the clone
//! still has the digest of the store it was cloned from.
//!
//! The paged storage has no digest cell: see
//! [`paged_deep_clone_stays_paged_with_its_codec`].

use super::core::KvCache;
use super::store_bytes_tests::{append, store_digest, CHUNK_SEQ, SHAPE_A, TEST_MAX_SEQ};
use crate::storage::KvStorage;
use crate::test_utils::{env_lock, f32_arr, lcg_data, TEST_SEED};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::{Array, Device};

/// One-token appends after the chunk, so the store holds more than one block.
const DECODE_STEPS: u64 = 2;

fn rows(seq: i32, seed: u64) -> (Array, Array) {
    let (kv_h, head_dim) = SHAPE_A;
    let shape = [1_i32, kv_h, seq, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    (
        f32_arr(&lcg_data(n, seed), &shape),
        f32_arr(&lcg_data(n, seed ^ 0x5a5a), &shape),
    )
}

/// Append one chunk and [`DECODE_STEPS`] single tokens with `in_prefill`
/// false.
pub(super) fn fill(cache: &mut KvCache, quant: KvQuant) {
    let mut sink = Vec::new();
    let (k, v) = rows(CHUNK_SEQ, TEST_SEED);
    append(cache, quant, &k, &v, &mut sink, Device::Cpu);
    for step in 1..=DECODE_STEPS {
        let (k, v) = rows(1, TEST_SEED.wrapping_add(step));
        append(cache, quant, &k, &v, &mut sink, Device::Cpu);
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: a clone of a store this test just filled must succeed, and the panic names the codec"
)]
fn deep_clone_keeps_every_store_byte_of_every_codec() {
    // The rotor K stores read the QJL switch when they are built.
    let _guard = env_lock();
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_MAX_SEQ);
        fill(&mut cache, quant);
        let source = store_digest(&cache.storage);
        let clone = cache.storage.try_deep_clone().expect("try_deep_clone");
        assert_eq!(
            store_digest(&clone),
            source,
            "{quant}: the deep clone does not hold the bytes of the store it was cloned from"
        );

        let mut sink = Vec::new();
        let (k, v) = rows(1, TEST_SEED ^ 0xc10e);
        append(&mut cache, quant, &k, &v, &mut sink, Device::Cpu);
        assert_eq!(
            store_digest(&clone),
            source,
            "{quant}: an append to the source moved the clone, so the clone shares a buffer"
        );
    }
}

/// A paged storage clones to a paged storage of the same codec and `max_seq`.
///
/// Not a digest cell: the paged update writes pages only on `Device::Gpu` and
/// falls back to the bf16 seed on the CPU, so no CPU fill has pages to clone.
/// The storage is built directly, because `KvStorage::new` builds it only when
/// the CLI latched the paged switch, which no test in this binary can do.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: a clone of an empty paged storage must succeed, and the panic names the codec"
)]
fn paged_deep_clone_stays_paged_with_its_codec() {
    for quant in [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::Planar3,
    ] {
        let storage = KvStorage::Paged {
            quant,
            k: None,
            v_k8: None,
            v_planar: None,
            max_seq: TEST_MAX_SEQ,
        };
        let clone = storage.try_deep_clone().expect("try_deep_clone");
        let KvStorage::Paged {
            quant: clone_quant,
            max_seq,
            ..
        } = clone
        else {
            panic!("{quant}: the clone of a paged storage is not paged");
        };
        assert_eq!(
            clone_quant, quant,
            "{quant}: the paged clone changed its codec"
        );
        assert_eq!(
            max_seq, TEST_MAX_SEQ,
            "{quant}: the paged clone changed max_seq"
        );
    }
}
