//! A hydrated layer holds `None` storage or the same storage variant its own
//! codec builds.
//!
//! Each hydrated layer gets the codec the arch builder gives that layer
//! (`ArchPromptCache::layer_quants` in `rmlx-models`), not the block's base
//! codec. `KvCache::update` takes its
//! entry from the storage and `exit_prefill` from the codec. The two keys agree
//! when a non-`None` hydrated storage has the same storage variant
//! `KvStorage::new` builds for that codec. The test compares the variant only,
//! not the widths or parameters it carries. A layer whose codec builds no packed
//! store spills geometry only and comes back as `None`.
//!
//! This test holds that for a layer spilled and hydrated under its own codec.
//! `ssd_boundary_codec_tests` in `rmlx-models` holds it for a boundary layer
//! whose codec differs from the base.

use super::block_io_tests::{arr, lcg};
use super::{read_caches, write_caches};
use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::storage::KvStorage;
use rmlx_kv_quant::{KvCache, KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::Device;
use std::mem::discriminant;
use tempfile::TempDir;

const MODEL_ID: &str = "Qwen3ForCausalLM/storage-family";
const MAX_SEQ: i32 = 256;
const PREFILL_SEQ: i32 = 24;

/// One layer at `quant`, driven through the prefill bracket: the state a
/// prompt-cache spill writes.
#[allow(
    clippy::expect_used,
    reason = "test driver: every spelling accepts this shape, so a failure is the defect under test"
)]
fn prefilled(quant: KvQuant) -> KvCache {
    let device = Device::Cpu;
    let chunk = [1_i32, 2, PREFILL_SEQ, 128];
    let n: usize = chunk.iter().map(|&d| d as usize).product();
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);
    cache.enter_prefill();
    cache
        .update(
            &arr(&lcg(n, 0x61), &chunk),
            &arr(&lcg(n, 0x62), &chunk),
            device,
        )
        .expect("prefill chunk");
    cache.exit_prefill(device).expect("exit_prefill");
    cache
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: a cache this test just built spills and reads back, so a failure is the defect under test"
)]
fn a_hydrated_layer_holds_none_or_the_storage_of_its_codec() {
    let device = Device::Cpu;
    let dir = TempDir::new().expect("temp dir");
    for (i, &quant) in ALL_KV_QUANTS.iter().enumerate() {
        let path = dir.path().join(format!("family_{i}.safetensors"));
        write_caches(&path, device, MODEL_ID, quant, &[prefilled(quant)], &[]).expect("spill");
        let (hydrated, _lin) = read_caches(
            &path,
            device,
            MODEL_ID,
            quant,
            &[quant],
            DispatchPolicy::default(),
            false,
        )
        .expect("hydrate");
        let [layer] = hydrated.as_slice() else {
            panic!("{quant}: one layer spilled, {} hydrated", hydrated.len());
        };
        let storage = layer.storage();
        let is_none = discriminant(storage) == discriminant(&KvStorage::None {});
        assert_eq!(
            is_none,
            !quant.materialises_packed_store(),
            "{quant}: a hydrated layer is None storage exactly when its codec builds no packed store"
        );
        if !is_none {
            assert!(
                discriminant(storage) == discriminant(&KvStorage::new(quant)),
                "{quant}: the hydrated storage is not the same storage variant this codec builds"
            );
        }
    }
}
