//! Unit tests for [`KvCache::resident_bytes`].
//!
//! All tests run on `Device::Cpu` — no GPU required.  They verify that
//! `resident_bytes()` returns the actual allocated buffer size, not a formula
//! estimate, and that empty (unpopulated) caches return 0.
//!
//! The GPU-backed quantised paths (K8V4, TurboSym, etc.) are exercised in the
//! integration test suite under `crates/rmlx-kv-quant/tests/` where a Metal
//! GPU is available.  The CPU-level invariants tested here are:
//!
//! 1. A freshly constructed cache → `resident_bytes() == 0`.
//! 2. After manually populating `decode_fp16_k/v` → bytes match `shape ×
//!    itemsize` exactly.
//! 3. Two independently populated layers sum correctly.
//! 4. `KvStorage::None` contributes 0; only `KvCache::decode_fp16_k/v` are
//!    counted.
//!
//! Also pins the **bf16 store-boundary floor** (the model-agnostic f32-KV
//! guard): an f32 K/V fed through the prefill-seed and decode-store paths must
//! land in the cache as bf16 (2 B/elem), so a future upstream f32 leak fails
//! fast on the bytes-per-element invariant instead of silently doubling KV.

use super::core::KvCache;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

// ── helpers ────────────────────────────────────────────────────────────────

/// Build a `[B, kv_h, seq, D]` fp16 Array filled with zeros on CPU.
#[allow(
    clippy::unwrap_used,
    reason = "test helper — panic on failure is the desired outcome"
)]
fn bf16_zeros(b: i32, kv_h: i32, seq: i32, d: i32) -> Array {
    let n = (b * kv_h * seq * d) as usize;
    // bf16 = 2 bytes per element; zero-fill.
    let bytes = vec![0u8; n * 2];
    Array::from_bytes(&bytes, &[b, kv_h, seq, d], Dtype::Bf16).unwrap()
}

/// Build a `[B, kv_h, seq, D]` **f32** Array with small non-zero values on CPU.
///
/// Models the f32-KV leak: an upstream f32 scalar promoted the attention stream
/// to f32, so the cache receives f32 (4 B/elem) K/V at the store boundary.
#[allow(
    clippy::unwrap_used,
    reason = "test helper — panic on failure is the desired outcome"
)]
fn f32_ramp(b: i32, kv_h: i32, seq: i32, d: i32) -> Array {
    let n = (b * kv_h * seq * d) as usize;
    let data: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.01).collect();
    Array::from_f32_slice(&data, &[b, kv_h, seq, d]).unwrap()
}

// ── tests ──────────────────────────────────────────────────────────────────

/// A freshly constructed `KvCache` has no buffers → `resident_bytes` is 0.
#[test]
fn fresh_cache_is_zero() {
    let cache = KvCache::with_quant_max_seq(KvQuant::None, 512);
    assert_eq!(
        cache.resident_bytes(),
        0,
        "freshly constructed cache must report 0 resident bytes"
    );
}

/// A K8V4 cache (quantised variant) also starts at 0 before any generate.
#[test]
fn fresh_k8v4_cache_is_zero() {
    let cache = KvCache::with_quant_max_seq(KvQuant::K8V4, 512);
    assert_eq!(
        cache.resident_bytes(),
        0,
        "pre-generate K8V4 cache must report 0 resident bytes"
    );
}

/// After injecting a bf16 K seed into `decode_fp16_k`, the bytes reported
/// equal `B × kv_h × seq × head_dim × 2` (bf16 = 2 bytes/elem).
///
/// `resident_bytes` must read the actual Array shape, not a stored formula.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn none_quant_with_seed_reports_actual_bytes() {
    const B: i32 = 1;
    const KV_H: i32 = 8;
    const SEQ: i32 = 64;
    const D: i32 = 128;

    // Build the cache and plant a bf16 K seed directly into decode_fp16_k.
    // We use the helper method that the prefill exit path uses internally
    // (`inject_fp16_seeds_for_test` isn't public, so we round-trip through
    // a minimal prefill instead).

    // Allocate a raw K/V pair, then force the cache into a populated-seed
    // state by running a minimal CPU prefill + exit_prefill.
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 512);
    let device = Device::Cpu;

    // Construct a dummy K/V prefill block (1 batch, KV_H heads, SEQ tokens, D dims).
    let k = bf16_zeros(B, KV_H, SEQ, D);
    let v = bf16_zeros(B, KV_H, SEQ, D);

    // enter_prefill / update / exit_prefill to populate decode_fp16_k/v.
    cache.enter_prefill();
    cache
        .update(&k, &v, device)
        .expect("update must succeed on CPU for KvQuant::None");
    cache
        .exit_prefill(device)
        .expect("exit_prefill must succeed on CPU for KvQuant::None");

    // For KvQuant::None the bf16 seed is stored on decode_fp16_k/v.
    // Expected: 2 buffers × B × kv_h × seq × head_dim × 2 bytes.
    let expected = 2 * B as u64 * KV_H as u64 * SEQ as u64 * D as u64 * 2;
    assert_eq!(
        cache.resident_bytes(),
        expected,
        "resident_bytes after prefill must equal 2 × B × kv_h × seq × D × 2 (bf16)"
    );
}

/// `resident_bytes` grows proportionally when the sequence length doubles.
/// This pins the linear relationship: bytes ∝ seq.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn resident_bytes_scales_with_seq() {
    const B: i32 = 1;
    const KV_H: i32 = 4;
    const D: i32 = 64;
    let device = Device::Cpu;

    let run = |seq: i32| -> u64 {
        let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 512);
        let k = bf16_zeros(B, KV_H, seq, D);
        let v = bf16_zeros(B, KV_H, seq, D);
        cache.enter_prefill();
        cache.update(&k, &v, device).unwrap();
        cache.exit_prefill(device).unwrap();
        cache.resident_bytes()
    };

    let bytes_32 = run(32);
    let bytes_64 = run(64);
    assert_eq!(
        bytes_64,
        bytes_32 * 2,
        "resident_bytes must scale linearly with seq (64 = 2 × 32)"
    );
    // Absolute value: 2 bufs × B=1 × kv_h=4 × seq=64 × D=64 × 2 bytes (bf16) = 65_536.
    assert_eq!(
        bytes_64, 65_536_u64,
        "absolute byte count for seq=64 must be exact (2 × 1 × 4 × 64 × 64 × 2)"
    );
}

/// When the bf16 decode buffer is over-allocated to a ceiling capacity larger
/// than the filled length (the on-device case: the mirror is sized to
/// max-context but only `offset` positions hold live K/V), `resident_bytes`
/// must report only the *filled* prefix — not the whole ceiling allocation.
///
/// This is the live-inference-KV accounting fix: a ceiling-sized buffer must
/// not inflate the reported KV (e.g. an 8192-capacity buffer holding 4096 live
/// positions reports half the bytes a naive shape × dtype sum would).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn ceiling_sized_decode_buffer_counts_filled_prefix_only() {
    const B: i32 = 1;
    const KV_H: i32 = 2;
    const D: i32 = 64;
    const CAPACITY: i32 = 8192; // allocated ceiling
    const FILLED: i32 = 4096; // live positions

    // Plant K/V buffers allocated to CAPACITY but only FILLED positions live.
    let k = bf16_zeros(B, KV_H, CAPACITY, D);
    let v = bf16_zeros(B, KV_H, CAPACITY, D);
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, CAPACITY);
    cache.inject_decode_fp16_for_test(k, v, FILLED);

    // Expected: 2 buffers × B × kv_h × FILLED × D × 2 bytes — NOT CAPACITY.
    let expected_filled = 2 * B as u64 * KV_H as u64 * FILLED as u64 * D as u64 * 2;
    let naive_capacity = 2 * B as u64 * KV_H as u64 * CAPACITY as u64 * D as u64 * 2;
    assert_eq!(
        cache.resident_bytes(),
        expected_filled,
        "resident_bytes must count only the filled prefix, not the ceiling capacity"
    );
    assert!(
        cache.resident_bytes() < naive_capacity,
        "filled-prefix count must be strictly less than the ceiling-capacity sum \
         (the inflation the live-KV accounting fix removes)"
    );
}

/// Two separately populated `KvCache` instances (simulating two model
/// layers) sum correctly.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn layer_sum_is_additive() {
    const B: i32 = 1;
    const KV_H: i32 = 4;
    const SEQ: i32 = 32;
    const D: i32 = 64;
    let device = Device::Cpu;

    let make_cache = || -> KvCache {
        let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 256);
        let k = bf16_zeros(B, KV_H, SEQ, D);
        let v = bf16_zeros(B, KV_H, SEQ, D);
        cache.enter_prefill();
        cache.update(&k, &v, device).unwrap();
        cache.exit_prefill(device).unwrap();
        cache
    };

    let layer0 = make_cache();
    let layer1 = make_cache();
    let per_layer = layer0.resident_bytes();
    assert!(
        per_layer > 0,
        "per-layer bytes must be non-zero after prefill"
    );

    let total: u64 = [layer0, layer1].iter().map(KvCache::resident_bytes).sum();
    assert_eq!(
        total,
        per_layer * 2,
        "sum across two identical layers must equal 2 × per-layer"
    );
}

// ── bf16 store-boundary floor (model-agnostic f32-KV guard) ──────────────────
//
// These pin the cache-level bf16 floor: the unquantised (`KvQuant::None`) /
// warm-TTFT store boundary casts incoming K/V to bf16 regardless of the
// inbound dtype, so a future upstream f32 leak can never silently double
// resident KV. The detector is the stored-buffer dtype (2 B/elem) — if the
// cast is removed, an f32 input stores at 4 B/elem and these go RED.

/// f32 K/V fed through the **prefill seed** path (`update_prefill_raw` →
/// `exit_prefill`) is stored as bf16 — the seed dtype is floored independent of
/// the inbound f32 stream.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn f32_prefill_seed_is_stored_bf16() {
    const B: i32 = 1;
    const KV_H: i32 = 2;
    const SEQ: i32 = 16;
    const D: i32 = 64;
    let device = Device::Cpu;

    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 256);
    let k = f32_ramp(B, KV_H, SEQ, D);
    let v = f32_ramp(B, KV_H, SEQ, D);
    assert_eq!(k.dtype(), Dtype::F32, "input K is f32 (the leak case)");

    cache.enter_prefill();
    cache.update(&k, &v, device).unwrap();
    cache.exit_prefill(device).unwrap();

    let (sk, sv) = cache
        .decode_fp16_kv()
        .expect("None-path prefill seed must populate decode_fp16_k/v");
    assert_eq!(
        sk.dtype(),
        Dtype::Bf16,
        "f32 K must be floored to bf16 at the prefill store boundary"
    );
    assert_eq!(
        sv.dtype(),
        Dtype::Bf16,
        "f32 V must be floored to bf16 at the prefill store boundary"
    );

    // Bytes-per-element invariant: 2 buffers × B × kv_h × SEQ × D × 2 (bf16).
    let elems_per_buf = (B * KV_H * SEQ * D) as u64;
    let expected = 2 * elems_per_buf * 2;
    assert_eq!(
        cache.resident_bytes(),
        expected,
        "resident KV must be 2 B/elem (bf16), not 4 B/elem (f32 leak)"
    );
    // Explicit bytes/elem == 2 derivation, independent of the formula above.
    let bytes_per_elem = cache.resident_bytes() / (2 * elems_per_buf);
    assert_eq!(bytes_per_elem, 2, "stored KV must be 2 bytes per element");
}

/// f32 K/V fed through the **decode store** path (`update_decode_fp16`, the
/// post-`exit_prefill` per-step append) is stored as bf16 — the decode mirror
/// stays floored, not just the prefill seed.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn f32_decode_store_is_stored_bf16() {
    const B: i32 = 1;
    const KV_H: i32 = 2;
    const SEQ: i32 = 8;
    const D: i32 = 64;
    let device = Device::Cpu;

    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 256);
    // Prefill with bf16 so the decode-store path (not the prefill path) is the
    // one under test on the following f32 step.
    let pk = bf16_zeros(B, KV_H, SEQ, D);
    let pv = bf16_zeros(B, KV_H, SEQ, D);
    cache.enter_prefill();
    cache.update(&pk, &pv, device).unwrap();
    cache.exit_prefill(device).unwrap();

    // One decode step with an f32 K/V (the per-token leak case).
    let dk = f32_ramp(B, KV_H, 1, D);
    let dv = f32_ramp(B, KV_H, 1, D);
    assert_eq!(
        dk.dtype(),
        Dtype::F32,
        "decode-step K is f32 (the leak case)"
    );
    cache.update(&dk, &dv, device).unwrap();

    let (sk, sv) = cache
        .decode_fp16_kv()
        .expect("decode mirror must be populated after a decode step");
    assert_eq!(
        sk.dtype(),
        Dtype::Bf16,
        "f32 K must be floored to bf16 at the decode store boundary"
    );
    assert_eq!(
        sv.dtype(),
        Dtype::Bf16,
        "f32 V must be floored to bf16 at the decode store boundary"
    );
}

/// Control: a bf16 input stays bf16 (the floor is idempotent — the steady state
/// after the per-arch source fixes is a pure no-op).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn bf16_input_stays_bf16_no_op() {
    const B: i32 = 1;
    const KV_H: i32 = 2;
    const SEQ: i32 = 16;
    const D: i32 = 64;
    let device = Device::Cpu;

    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 256);
    let k = bf16_zeros(B, KV_H, SEQ, D);
    let v = bf16_zeros(B, KV_H, SEQ, D);
    cache.enter_prefill();
    cache.update(&k, &v, device).unwrap();
    cache.exit_prefill(device).unwrap();

    let (sk, sv) = cache.decode_fp16_kv().expect("seed must be populated");
    assert_eq!(sk.dtype(), Dtype::Bf16, "bf16 K stays bf16");
    assert_eq!(sv.dtype(), Dtype::Bf16, "bf16 V stays bf16");
}
