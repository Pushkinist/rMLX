//! A hydrated cache holds the capacity the spilled cache held.
//!
//! `KvStorage` carries no `max_seq`. The writer takes it from the cache, and
//! the reader hands the geometry's value to `KvCache::from_storage`. Both
//! layers below hold a capacity other than `KV_MAX_SEQ_DEFAULT`, so a reader
//! that drops the geometry's value reads red here.

use super::block_io_tests::{arr, lcg};
use super::{read_caches, write_caches};
use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::{KvCache, KvQuant, KV_MAX_SEQ_DEFAULT};
use rmlx_mlx::Device;
use tempfile::TempDir;

const MODEL_ID: &str = "Qwen3ForCausalLM/max-seq-pin";
const QUANT: KvQuant = KvQuant::K8V8;

/// Starting capacity of the store-backed layer, below the tokens it takes, so
/// the decode grow raises it before the spill.
const START_MAX_SEQ: i32 = 16;
const FILLED_SEQ: i32 = 24;
/// Capacity of the geometry-only layer.
const WIDE_MAX_SEQ: i32 = 8192;

#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: a cache this test just built spills and reads back, so a failure is the defect under test"
)]
fn hydrate_restores_the_max_seq_each_layer_spilled() {
    let device = Device::Cpu;
    let shape = [1_i32, 2, FILLED_SEQ, 128];
    let n: usize = shape.iter().map(|&d| d as usize).product();

    // Appended outside a prefill bracket, so the decode path writes the store
    // and grows the capacity.
    let mut filled = KvCache::with_quant_max_seq(QUANT, START_MAX_SEQ);
    filled
        .update(
            &arr(&lcg(n, 0x51), &shape),
            &arr(&lcg(n, 0x52), &shape),
            device,
        )
        .expect("append");
    let grown = filled.max_seq();
    assert!(
        grown > START_MAX_SEQ && grown != KV_MAX_SEQ_DEFAULT,
        "precondition: the store-backed layer grew to a capacity of its own, got {grown}"
    );
    assert!(
        !filled.storage().is_geometry_only(),
        "precondition: the first layer spills a packed store"
    );

    let empty = KvCache::with_quant_max_seq(QUANT, WIDE_MAX_SEQ);
    assert!(
        empty.storage().is_geometry_only(),
        "precondition: the second layer spills geometry only"
    );

    let dir = TempDir::new().expect("temp dir");
    let path = dir.path().join("max_seq.safetensors");
    write_caches(&path, device, MODEL_ID, QUANT, &[filled, empty], &[]).expect("spill");
    let (hydrated, _lin) = read_caches(
        &path,
        device,
        MODEL_ID,
        QUANT,
        DispatchPolicy::default(),
        false,
    )
    .expect("hydrate");

    let got: Vec<i32> = hydrated.iter().map(KvCache::max_seq).collect();
    assert_eq!(
        got,
        [grown, WIDE_MAX_SEQ],
        "each hydrated layer must hold the max_seq its geometry recorded"
    );
}
