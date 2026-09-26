//! What `KvCache::max_seq` holds after each path that builds, grows, clones,
//! cuts or resets a cache.
//!
//! Every codec is driven on the CPU and the capacity is pinned to a closed
//! form. The prefill grow and the decode grow are the only writers after
//! construction; a grow that does not write the field leaves the capacity at
//! its start value, and the pins below read that as a failure.

use super::core::KvCache;
use super::deep_clone_digest_tests::fill;
use super::store_bytes_tests::{CHUNK_SEQ, SHAPE_A};
use super::update::next_pow2_seq;
use crate::test_utils::{env_lock, f32_arr, lcg_data, TEST_SEED};
use crate::ALL_KV_QUANTS;
use rmlx_mlx::{Array, Device};

/// Starting capacity, below [`CHUNK_SEQ`] so that the first chunk grows it.
const START_MAX_SEQ: i32 = 16;

fn chunk(seq: i32) -> (Array, Array) {
    let (kv_h, head_dim) = SHAPE_A;
    let shape = [1_i32, kv_h, seq, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    (
        f32_arr(&lcg_data(n, TEST_SEED), &shape),
        f32_arr(&lcg_data(n, TEST_SEED ^ 0x5a5a), &shape),
    )
}

#[test]
fn construction_holds_the_requested_max_seq() {
    let _guard = env_lock();
    for &quant in ALL_KV_QUANTS {
        for max_seq in [1, START_MAX_SEQ, 8192] {
            let cache = KvCache::with_quant_max_seq(quant, max_seq);
            assert_eq!(cache.max_seq(), max_seq, "{quant}: construction");
        }
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test driver: every step runs on a shape each codec accepts, and the panic names the codec"
)]
fn the_prefill_grow_holds_through_exit_prefill_clone_truncate_and_reset() {
    // The rotor K stores read the QJL switch when they are built.
    let _guard = env_lock();
    let grown = next_pow2_seq(CHUNK_SEQ);
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, START_MAX_SEQ);
        cache.enter_prefill();
        let (k, v) = chunk(CHUNK_SEQ);
        cache.update(&k, &v, Device::Cpu).expect("prefill chunk");
        assert_eq!(cache.max_seq(), grown, "{quant}: the prefill grow");

        cache.exit_prefill(Device::Cpu).expect("exit_prefill");
        assert_eq!(cache.max_seq(), grown, "{quant}: exit_prefill");

        let clone = cache.try_deep_clone().expect("try_deep_clone");
        assert_eq!(clone.max_seq(), grown, "{quant}: deep clone");

        cache.truncate_to(CHUNK_SEQ / 2).expect("truncate_to");
        assert_eq!(cache.max_seq(), grown, "{quant}: truncate_to");

        cache.reset();
        assert_eq!(cache.max_seq(), grown, "{quant}: reset");
    }
}

/// The mixed pair decodes through `update_and_sdpa`, whose state grows in its
/// own steps and never asks for decode capacity, so its capacity stays where it
/// started. Every other codec grows to the next power of two of its offset.
#[test]
fn the_decode_grow_holds_the_grown_max_seq() {
    let _guard = env_lock();
    for &quant in ALL_KV_QUANTS {
        let mut cache = KvCache::with_quant_max_seq(quant, START_MAX_SEQ);
        fill(&mut cache, quant);
        let want = if quant.uses_mixed_path() {
            START_MAX_SEQ
        } else {
            next_pow2_seq(cache.offset())
        };
        assert_eq!(cache.max_seq(), want, "{quant}: the decode grow");
    }
}
