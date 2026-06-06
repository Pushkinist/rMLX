//! Dynamic grow + hard cap for `update_prefill_raw`.
//!
//! These tests pin two behaviours of the prefill-raw buffer:
//!
//! * **Dynamic grow** — when a prefill chunk would push `offset` past the
//!   currently allocated `[B, kv_h, max_seq, head_dim]` buffer, the buffer
//!   grows to the next power-of-two ≥ needed and the existing filled prefix
//!   is preserved.  Before the fix, the slice_update target
//!   `[prev_offset, new_offset]` collapsed to zero-width inside the original
//!   buffer once `prev_offset >= max_seq`, producing a broadcast-shape error
//!   along the lines of `(1, 2, 512, 512) vs (1, 2, 0, 512)`.
//!
//! * **Hard cap** — `RMLX_KV_MAX_SEQ_HARD_CAP` is an opt-in hard cap. Tests
//!   leave it unset by default; the cap path is verified at the unit level
//!   via [`super::update::kv_hard_cap`] (private helper, see
//!   integration-style smoke run for end-to-end). We do NOT set the env var
//!   from a parallel test because `OnceLock` makes the resolution
//!   process-global.
//!
//! Repro recipe (CLI):
//!   ./target/release/rmlx baseline --model <BONSAI_PATH> --kv-quant k8v4
//! Before the fix this failed once total prompt > 4096 tokens (default
//! `KV_MAX_SEQ_DEFAULT`).  After the fix it grows past 4096 and completes.

use super::core::KvCache;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and the bytes are
    // copied into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

const TEST_KV_H: i32 = 2;
const TEST_HEAD_DIM: i32 = 64;
// Use a tiny starting max_seq so we can exceed it cheaply in a unit test.
const TEST_INITIAL_MAX_SEQ: i32 = 128;
const TEST_CHUNK: i32 = 64;

/// Reproducer: prefill past `max_seq` must NOT panic / error.
///
/// Drives a `KvQuant::None` cache (bf16 path — simplest, no MSL kernels) on
/// `Device::Cpu` with a max_seq of 128 tokens, then prefills 3 × 64-token
/// chunks for a total of 192 tokens. The third chunk pushes `prev_offset`
/// past the original `max_seq=128`; pre-fix, this is the broadcast-shape
/// error.  Post-fix, the buffer grows to 256 and the chunk lands cleanly.
#[test]
fn update_prefill_raw_grows_past_initial_max_seq() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ);
    cache.enter_prefill();

    let chunk_shape = [1_i32, TEST_KV_H, TEST_CHUNK, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();

    // Three chunks: 64 + 64 + 64 = 192 > 128 (initial cap).
    for step in 0..3_u32 {
        let k = f32_arr(&vec![0.1_f32 * (step as f32 + 1.0); n_chunk], &chunk_shape);
        let v = f32_arr(&vec![0.2_f32 * (step as f32 + 1.0); n_chunk], &chunk_shape);
        let _ = cache
            .update(&k, &v, device)
            .unwrap_or_else(|e| panic!("prefill grow regression on chunk {step}: {e}"));
    }

    assert_eq!(
        cache.offset(),
        192,
        "offset should track total prefill length",
    );

    cache.exit_prefill(device).expect("exit_prefill");
}

/// A single chunk that overflows the initial buffer also works.
///
/// The grow path must trigger on the first call when the first chunk alone
/// is larger than the initial cap.  This pins the lazy-allocation branch of
/// `ensure_prefill_capacity` (no existing buffer to copy forward).
#[test]
fn update_prefill_raw_grows_on_first_oversized_chunk() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ);
    cache.enter_prefill();

    // 300 > 128 (initial cap). Next pow2 ≥ 300 = 512.
    let chunk_shape = [1_i32, TEST_KV_H, 300, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.5_f32; n_chunk], &chunk_shape);
    let v = f32_arr(&vec![0.7_f32; n_chunk], &chunk_shape);

    let (k_full, v_full) = cache
        .update(&k, &v, device)
        .expect("grow on first oversized chunk");

    assert_eq!(cache.offset(), 300);
    assert_eq!(k_full.shape()[2], 300);
    assert_eq!(v_full.shape()[2], 300);

    cache.exit_prefill(device).expect("exit_prefill");
}

/// Resume + grow is rejected with a typed error.
///
/// Drives a `KvQuant::None` cache (bf16, no MSL kernels) on CPU:
///   1. Enter prefill, push one chunk that fits in the initial 128-token cap.
///   2. Exit prefill — materialises the fp16 decode seed; the storage
///      `max_seq` stays at 128 with a populated `decode_fp16_k/v` pair.
///   3. Re-enter prefill (`enter_prefill` clears `prefill_raw_*` but does
///      NOT reset the storage payload).
///   4. Push a chunk large enough to exceed the current `max_seq=128`.
///
/// Pre-fix: `ensure_prefill_capacity` would silently bump the storage
/// `max_seq` to 256, leaving the existing `decode_fp16_k` shape disagreeing
/// with the new scalar — shape assert / silent truncation downstream.
/// Post-fix: the grow guard detects the materialised payload and returns
/// a typed error.
#[test]
fn update_prefill_raw_rejects_grow_on_resumed_cache() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ);

    // 1. First prefill — fits in 128 cap (64 tokens).
    cache.enter_prefill();
    let chunk_shape = [1_i32, TEST_KV_H, TEST_CHUNK, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.1_f32; n_chunk], &chunk_shape);
    let v = f32_arr(&vec![0.2_f32; n_chunk], &chunk_shape);
    cache.update(&k, &v, device).expect("first chunk");

    // 2. Exit — materialises fp16 decode seed for KvQuant::None.
    cache.exit_prefill(device).expect("exit_prefill");

    // 3. Re-enter — clears prefill_raw_* but leaves decode_fp16_k/v populated.
    cache.enter_prefill();

    // 4. Push a second chunk; combined offset = 64 + 96 = 160 > 128 cap. The
    //    grow path must detect the resumed cache and refuse.
    let oversize_seq = 96_i32;
    let oversize_shape = [1_i32, TEST_KV_H, oversize_seq, TEST_HEAD_DIM];
    let n_oversize: usize = oversize_shape.iter().map(|&d| d as usize).product();
    let k2 = f32_arr(&vec![0.3_f32; n_oversize], &oversize_shape);
    let v2 = f32_arr(&vec![0.4_f32; n_oversize], &oversize_shape);

    let err = cache
        .update(&k2, &v2, device)
        .expect_err("resumed-cache grow must error, not corrupt");
    let msg = err.to_string();
    assert!(
        msg.contains("grow not legal after exit_prefill"),
        "unexpected error: {msg}",
    );
}

/// `next_pow2_seq` rounds correctly for a representative spread.
///
/// Pins the local helper so future refactors don't silently regress to
/// linear growth.
#[test]
fn next_pow2_seq_rounds_up() {
    use super::update::next_pow2_seq;
    assert_eq!(next_pow2_seq(0), 1);
    assert_eq!(next_pow2_seq(1), 1);
    assert_eq!(next_pow2_seq(2), 2);
    assert_eq!(next_pow2_seq(3), 4);
    assert_eq!(next_pow2_seq(4097), 8192);
    assert_eq!(next_pow2_seq(10867), 16384);
    assert_eq!(next_pow2_seq(16384), 16384);
    assert_eq!(next_pow2_seq(1 << 30), 1 << 30);
    // Saturating clamp at 2^30.
    assert_eq!(next_pow2_seq(i32::MAX), 1 << 30);
}
