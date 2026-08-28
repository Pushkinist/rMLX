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

/// Read an array back as `f32`, whatever its stored dtype.
///
/// The prefill ring is stored bf16 (`cast_store_bf16` at the append boundary),
/// so a byte-level read needs the widening cast first.
fn read_f32(a: &Array, device: Device) -> Vec<f32> {
    let f = a.astype(Dtype::F32, device).expect("astype f32");
    f.eval().expect("eval");
    f.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact(4) yields 4 bytes")))
        .collect()
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

/// Issue #25: a large `max_seq_ceiling` does NOT pre-allocate the ring.
///
/// The ring must start at its small initial `max_seq` and grow lazily up to
/// the ceiling. Build a cache with a tiny initial `max_seq=128` and a large
/// ceiling of 140_000 (the issue's oversized `--max-ctx`), then prefill 192
/// tokens. The buffer must end up sized to the lazy power-of-two (256), NOT to
/// the ceiling — proving the short request pays only for what it fills.
#[test]
fn ceiling_does_not_pre_allocate_ring() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ)
        .with_max_seq_ceiling(140_000);
    cache.enter_prefill();

    let chunk_shape = [1_i32, TEST_KV_H, TEST_CHUNK, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    for step in 0..3_u32 {
        let k = f32_arr(&vec![0.1_f32 * (step as f32 + 1.0); n_chunk], &chunk_shape);
        let v = f32_arr(&vec![0.2_f32 * (step as f32 + 1.0); n_chunk], &chunk_shape);
        cache
            .update(&k, &v, device)
            .expect("lazy grow under ceiling");
    }

    assert_eq!(cache.offset(), 192, "offset tracks filled length");
    // The ring grew only to the next pow2 ≥ 192 = 256 — NOT to 140_000.
    assert_eq!(
        cache.storage_max_seq_for_test(),
        256,
        "ring grows lazily to the doubling boundary, not the ceiling",
    );

    cache.exit_prefill(device).expect("exit_prefill");
}

/// Issue #25: lazy grow is clamped to the ceiling, never past it.
///
/// With initial `max_seq=128` and ceiling=200, a 192-token prefill would
/// normally double to 256, but the ceiling clamps the allocation to exactly
/// 200. The request still fits (192 <= 200) and completes.
#[test]
fn grow_clamps_to_ceiling() {
    let device = Device::Cpu;
    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ).with_max_seq_ceiling(200);
    cache.enter_prefill();

    let chunk_shape = [1_i32, TEST_KV_H, TEST_CHUNK, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    for _ in 0..3_u32 {
        let k = f32_arr(&vec![0.1_f32; n_chunk], &chunk_shape);
        let v = f32_arr(&vec![0.2_f32; n_chunk], &chunk_shape);
        cache.update(&k, &v, device).expect("lazy grow to ceiling");
    }

    assert_eq!(cache.offset(), 192);
    assert_eq!(
        cache.storage_max_seq_for_test(),
        200,
        "doubled size (256) clamped down to the 200-token ceiling",
    );

    cache.exit_prefill(device).expect("exit_prefill");
}

/// Issue #25: a prefill that exceeds the ceiling is rejected with a typed
/// error before any allocation past the ceiling.
///
/// Initial `max_seq=128`, ceiling=150. A single 200-token chunk needs more
/// than the ceiling and must error with `KvCeilingExceeded` rather than grow.
#[test]
fn prefill_over_ceiling_is_rejected() {
    let device = Device::Cpu;
    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ).with_max_seq_ceiling(150);
    cache.enter_prefill();

    let chunk_shape = [1_i32, TEST_KV_H, 200, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.5_f32; n_chunk], &chunk_shape);
    let v = f32_arr(&vec![0.7_f32; n_chunk], &chunk_shape);

    let err = cache
        .update(&k, &v, device)
        .expect_err("prefill over ceiling must error");
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds max-ctx ceiling"),
        "unexpected error: {msg}",
    );
}

/// A zero/negative ceiling is treated as "no ceiling" — unbounded lazy grow.
#[test]
fn non_positive_ceiling_means_unbounded() {
    let device = Device::Cpu;
    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::None, TEST_INITIAL_MAX_SEQ).with_max_seq_ceiling(0);
    cache.enter_prefill();

    // 300 > 128 initial cap; with no ceiling it grows to 512 as usual.
    let chunk_shape = [1_i32, TEST_KV_H, 300, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.5_f32; n_chunk], &chunk_shape);
    let v = f32_arr(&vec![0.7_f32; n_chunk], &chunk_shape);
    cache
        .update(&k, &v, device)
        .expect("unbounded grow with non-positive ceiling");

    assert_eq!(cache.offset(), 300);
    assert_eq!(cache.storage_max_seq_for_test(), 512);

    cache.exit_prefill(device).expect("exit_prefill");
}

/// The slots of the prefill ring no chunk ever writes hold zero, and attention
/// never sees them.
///
/// A prompt whose length is not a power of two leaves `max_seq - offset` slots
/// unwritten for the whole request. Two properties keep those slots out of the
/// numerics, and both are pinned here, because violating either one feeds
/// whatever the allocator last left in that page into SDPA — and a single NaN
/// or Inf reaching a softmax numerator turns the entire logit row into NaN:
///
/// 1. the buffer is allocated with `zeros()`, so an unwritten slot reads 0.0
///    rather than recycled device memory; and
/// 2. `update_prefill_raw` hands back `[.., 0..offset, ..]`, so the tail is not
///    part of the K/V the attention call receives at any prompt length.
///
/// Property 2 alone would be enough for correctness; property 1 is what makes
/// an accidental over-slice benign instead of non-deterministic, so losing
/// either is worth a failing test.
#[test]
fn prefill_ring_tail_is_zeroed_and_never_returned() {
    let device = Device::Cpu;
    // 100 tokens into a 128-slot ring: 28 slots stay unwritten.
    let filled = 100_i32;
    let max_seq = TEST_INITIAL_MAX_SEQ;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, max_seq);
    cache.enter_prefill();

    let chunk_shape = [1_i32, TEST_KV_H, filled, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    // 1.5 / 2.5 are exact in bf16, so the store's cast is lossless and
    // "written" vs "never written" stays a bit-exact distinction.
    let k = f32_arr(&vec![1.5_f32; n_chunk], &chunk_shape);
    let v = f32_arr(&vec![2.5_f32; n_chunk], &chunk_shape);

    let (k_full, v_full) = cache.update(&k, &v, device).expect("prefill chunk");

    // Property 2 — the attention input is the filled prefix, not the ring.
    assert_eq!(
        k_full.shape()[2],
        filled,
        "K handed to attention must span the filled prefix, not the ring capacity",
    );
    assert_eq!(
        v_full.shape()[2],
        filled,
        "V handed to attention must span the filled prefix, not the ring capacity",
    );
    assert!(
        read_f32(&k_full, device).iter().all(|&x| x == 1.5),
        "every returned K element is a written one",
    );
    assert!(
        read_f32(&v_full, device).iter().all(|&x| x == 2.5),
        "every returned V element is a written one",
    );

    // Property 1 — the unwritten remainder of the ring is zero.
    let ring = cache
        .prefill_raw_k
        .as_ref()
        .expect("prefill ring allocated on first append");
    assert_eq!(
        ring.shape()[2],
        max_seq,
        "precondition: the ring is larger than the filled prefix",
    );
    let vals = read_f32(ring, device);
    let row = TEST_HEAD_DIM as usize;
    let cap = max_seq as usize;
    let mut tail_dirty = 0_usize;
    for h in 0..TEST_KV_H as usize {
        for t in (filled as usize)..cap {
            let base = (h * cap + t) * row;
            let slot = vals
                .get(base..base + row)
                .expect("ring slot inside the readback");
            // `!= 0.0` and not a max-of-abs fold: `f32::max` propagates the
            // *other* operand past a NaN, so a fold would report a clean tail
            // for exactly the payload this test exists to catch.
            tail_dirty += slot.iter().filter(|&&x| x != 0.0).count();
        }
    }
    assert_eq!(
        tail_dirty, 0,
        "the never-written ring tail must be zero, not recycled device memory",
    );
}

/// Prefill append is codec-independent: the K/V attention receives is the same
/// under `KvQuant::None` and `KvQuant::K8V8`.
///
/// While `in_prefill` is set, `KvCache::update` routes every non-rotating cache
/// to `update_prefill_raw`; the codec dispatch sits behind that branch and no
/// quantisation happens until `exit_prefill`. So a fault observed *during*
/// prefill reproducing "at both k8v8 and none" says nothing about the codec —
/// the two cells execute the same appends over the same bf16 buffer. Pinned so
/// that a change which started quantising per chunk cannot silently make that
/// inference valid again while every other test stays green.
#[test]
fn prefill_append_is_codec_independent() {
    let device = Device::Cpu;
    let chunk_shape = [1_i32, TEST_KV_H, TEST_CHUNK, TEST_HEAD_DIM];
    let n_chunk: usize = chunk_shape.iter().map(|&d| d as usize).product();
    // A varied pattern, every value exact in bf16 (quarter steps in [-2, 2)):
    // a codec running here would round it and the comparison would separate.
    let pattern: Vec<f32> = (0..n_chunk).map(|i| (i % 16) as f32 * 0.25 - 2.0).collect();
    let k = f32_arr(&pattern, &chunk_shape);
    let v = f32_arr(&pattern, &chunk_shape);

    let mut out = Vec::new();
    for quant in [KvQuant::None, KvQuant::K8V8] {
        let mut cache = KvCache::with_quant_max_seq(quant, TEST_INITIAL_MAX_SEQ);
        cache.enter_prefill();
        let (k_full, v_full) = cache.update(&k, &v, device).expect("prefill chunk");
        out.push((read_f32(&k_full, device), read_f32(&v_full, device)));
    }

    let (none_k, none_v) = out.first().expect("None arm recorded");
    let (k8v8_k, k8v8_v) = out.get(1).expect("K8V8 arm recorded");
    assert_eq!(none_k, k8v8_k, "prefill K must not depend on the KV codec");
    assert_eq!(none_v, k8v8_v, "prefill V must not depend on the KV codec");
    assert_eq!(
        none_k, &pattern,
        "prefill K must be the unquantised input (bf16-exact pattern)",
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

/// The resumed-cache grow refusal survives eliding the `Mixed` bf16 mirror.
///
/// `storage_has_materialised_payload` used to answer for a mirrored `Mixed`
/// cache on its first clause — "a decode mirror exists" — and only fell through
/// to `state.offset > 0` when there was none. On a dense architecture there is
/// now never a mirror, so that second clause is the whole test, and it has to
/// hold on its own: a cache that has been through `exit_prefill` must still
/// refuse a grow rather than resize a buffer its packed payload was sized
/// against.
///
/// Both halves are asserted, because a guard that stopped refusing and a guard
/// that started refusing everything both "changed behaviour": a cache that has
/// only *entered* prefill is in the window where the grow is legal, and must
/// still be allowed to grow.
#[test]
fn dense_mixed_cache_still_refuses_a_grow_after_exit_prefill() {
    let device = Device::Cpu;
    let quant = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let chunk = |seq: i32, fill: f32| -> (Array, Array) {
        let shape = [1_i32, TEST_KV_H, seq, TEST_HEAD_DIM];
        let n: usize = shape.iter().map(|&d| d as usize).product();
        (
            f32_arr(&vec![fill; n], &shape),
            f32_arr(&vec![fill + 0.1; n], &shape),
        )
    };

    // A dense cache: no cross-layer KV sharing, so `exit_prefill` builds the
    // packed store and no mirror.
    let mut cache = KvCache::with_quant_max_seq(quant, TEST_INITIAL_MAX_SEQ).with_shares_kv(false);
    cache.enter_prefill();
    let (k, v) = chunk(TEST_CHUNK, 0.1);
    cache.update(&k, &v, device).expect("first chunk");
    cache.exit_prefill(device).expect("exit_prefill");
    assert!(
        cache.decode_fp16_k_for_test().is_none() && cache.decode_fp16_v_for_test().is_none(),
        "precondition: the dense arm keeps no mirror, so the guard cannot key on one"
    );

    cache.enter_prefill();
    let (k2, v2) = chunk(96, 0.3);
    let err = cache
        .update(&k2, &v2, device)
        .expect_err("resumed-cache grow must error, not resize under a live payload");
    assert!(
        err.to_string()
            .contains("grow not legal after exit_prefill"),
        "expected the resumed-cache grow refusal, got: {err}"
    );

    // Converse: before any `exit_prefill` there is no payload to disagree with,
    // and the grow is the legal lazy-ring expansion.
    let mut fresh = KvCache::with_quant_max_seq(quant, TEST_INITIAL_MAX_SEQ).with_shares_kv(false);
    fresh.enter_prefill();
    let (k3, v3) = chunk(TEST_CHUNK, 0.5);
    fresh.update(&k3, &v3, device).expect("first chunk");
    let (k4, v4) = chunk(96, 0.7);
    fresh
        .update(&k4, &v4, device)
        .expect("a cache that has not exited prefill must still be allowed to grow");
    assert_eq!(fresh.offset(), TEST_CHUNK + 96);
}
