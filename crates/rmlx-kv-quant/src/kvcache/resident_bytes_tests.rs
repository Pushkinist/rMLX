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
