//! An update writes its K/V rows into the buffer that the cache already holds.
//!
//! MLX updates a buffer in place only when nothing else owns the old buffer at
//! the evaluation. Otherwise it allocates a new buffer and copies the whole
//! cache, at every step. Each step here is evaluated before the next step
//! starts, so no unfinished work holds the old buffer: a buffer address that
//! moves between steps names an owner that the update path kept. See
//! `docs/KV_UPDATE_PATH.md`, "In-place update".

use super::core::KvCache;
use crate::test_utils::{env_lock, f32_arr, lcg_data, TEST_SEED};
use crate::{KvQuant, ALL_KV_QUANTS};
use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device};

const KV_H: i32 = 2;
const HEAD_DIM: i32 = 64;
const MAX_SEQ: i32 = 256;
const PREFILL_SEQ: i32 = 8;
const STEPS: u64 = 6;
/// Smaller than the ring's growth step, so the ring allocates once and then
/// wraps: 40 steps write every slot more than twice.
const WINDOW: i32 = 16;
const RING_STEPS: u64 = 40;

/// K and V rows for `seq` positions.
fn rows(seq: i32, seed: u64) -> (Array, Array) {
    let shape = [1, KV_H, seq, HEAD_DIM];
    let n = (KV_H * seq * HEAD_DIM) as usize;
    (
        f32_arr(&lcg_data(n, seed), &shape),
        f32_arr(&lcg_data(n, seed ^ 0x5a5a), &shape),
    )
}

/// Call `update` `steps` times with `seq`-position rows. Return the K and V
/// buffer addresses after each call. Every row that `update` returns starts at
/// position 0 of its buffer, so its address is the address of the buffer. The
/// returned rows are dropped before the next call, because they share the
/// buffer.
fn addresses_per_step(cache: &mut KvCache, seq: i32, steps: u64) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for i in 0..steps {
        let (k, v) = rows(seq, TEST_SEED.wrapping_add(i + 1));
        let (k_rows, v_rows) = cache.update(&k, &v, Device::Cpu)?;
        out.push((k_rows.data_address()?, v_rows.data_address()?));
    }
    Ok(out)
}

/// One prefill chunk, `exit_prefill`, then [`STEPS`] decode steps.
fn decode_addresses(quant: KvQuant) -> Result<Vec<(usize, usize)>> {
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);
    cache.enter_prefill();
    let (k, v) = rows(PREFILL_SEQ, TEST_SEED);
    cache.update(&k, &v, Device::Cpu)?;
    cache.exit_prefill(Device::Cpu)?;
    addresses_per_step(&mut cache, 1, STEPS)
}

/// Every address in `addresses` is the first one.
fn one_buffer(addresses: &[usize]) -> bool {
    addresses.windows(2).all(|w| w[0] == w[1])
}

/// Every codec whose decode reads the bf16 mirror writes each decode row into
/// the mirror it holds, on each axis it reads from the mirror, on a layer that
/// does not share its KV. The `Mixed` / `RotK` mirror of a shared-KV producer
/// is not driven here.
#[test]
fn every_decode_step_writes_into_the_mirror_it_holds() {
    let _guard = env_lock();
    let mut checked = Vec::new();
    for &quant in ALL_KV_QUANTS {
        let k_mirror = quant.feeds_bf16_k_at_decode(false);
        let v_mirror = quant.feeds_bf16_v_at_decode(false);
        if !k_mirror && !v_mirror {
            continue;
        }
        let steps = match decode_addresses(quant) {
            Ok(steps) => steps,
            Err(e) => panic!("{quant}: the drive failed: {e}"),
        };
        let (k_addresses, v_addresses): (Vec<usize>, Vec<usize>) = steps.into_iter().unzip();
        if k_mirror {
            assert!(
                one_buffer(&k_addresses),
                "{quant}: the K mirror moved between decode steps ({k_addresses:x?}). The \
                 update copied it: something still owned the old buffer when it was evaluated"
            );
        }
        if v_mirror {
            assert!(
                one_buffer(&v_addresses),
                "{quant}: the V mirror moved between decode steps ({v_addresses:x?}). The \
                 update copied it: something still owned the old buffer when it was evaluated"
            );
        }
        checked.push(quant);
    }
    assert!(
        checked.contains(&KvQuant::None),
        "the bf16 cache must be among the checked codecs; checked: {checked:?}"
    );
}

/// Each prefill chunk writes into the raw prefill buffer that the cache holds.
#[test]
fn every_prefill_chunk_writes_into_the_raw_buffer_it_holds() {
    let _guard = env_lock();
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ);
    cache.enter_prefill();
    let steps = match addresses_per_step(&mut cache, PREFILL_SEQ, STEPS) {
        Ok(steps) => steps,
        Err(e) => panic!("the prefill drive failed: {e}"),
    };
    let (k_addresses, v_addresses): (Vec<usize>, Vec<usize>) = steps.into_iter().unzip();
    assert!(
        one_buffer(&k_addresses) && one_buffer(&v_addresses),
        "the raw prefill buffers moved between chunks (K {k_addresses:x?}, V {v_addresses:x?}). \
         The update copied them: something still owned the old buffer when it was evaluated"
    );
}

/// Each decode step of a sliding-window layer writes into the ring that the
/// cache holds, also after the ring wraps.
#[test]
fn every_ring_step_writes_into_the_ring_it_holds() {
    let _guard = env_lock();
    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::None, MAX_SEQ, Some(WINDOW));
    let steps = match addresses_per_step(&mut cache, 1, RING_STEPS) {
        Ok(steps) => steps,
        Err(e) => panic!("the ring drive failed: {e}"),
    };
    let (k_addresses, v_addresses): (Vec<usize>, Vec<usize>) = steps.into_iter().unzip();
    assert!(
        one_buffer(&k_addresses) && one_buffer(&v_addresses),
        "the ring buffers moved between decode steps (K {k_addresses:x?}, V {v_addresses:x?}). \
         The update copied them: something still owned the old buffer when it was evaluated"
    );
}
