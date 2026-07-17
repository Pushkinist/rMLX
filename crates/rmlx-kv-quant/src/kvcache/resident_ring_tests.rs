//! `resident_bytes` sees the GPU ring that the ring-backed K codecs allocate.
//!
//! # Why this exists
//!
//! `k_iso3` / `k_rotor3` keep their CPU blocks *and* stand up a GPU-resident
//! ring on the first flash-decode dispatch. The ring is real memory — tens of
//! MB per layer at long context — but for a while the byte total counted the
//! CPU blocks alone, so allocating it moved the reported KV bytes by exactly
//! zero. That is worse than a wrong number: KV bytes is the axis these codecs
//! are accepted or rejected on, so an invisible allocation silently invalidates
//! the comparison.
//!
//! # What these prove, and what they do not
//!
//! These pin the ring into the total: standing one up must move
//! `resident_bytes` by at least the ring's own reported size, across both
//! ring-backed codecs and two contexts. That is the bug that was shipped — a
//! delta of exactly zero.
//!
//! They do **not** validate the ring's *magnitude*. The anchor
//! (`QuantKGpuRing::byte_size`, via `live_ring_bytes`) is part of the
//! accounting under test — `QuantIsoK3::byte_size` is literally
//! `blocks + gpu.byte_size()` — so a ring size that was uniformly wrong by a
//! constant factor would still pass here. Magnitude is anchored elsewhere, by
//! independent literals: `kv_cache/tests.rs`
//! (`resident_bytes_none_quant_is_the_two_bf16_mirrors`,
//! `ring_bytes_match_independent_geometry` below) and, at the whole-process
//! level, the RSS cross-check recorded in `docs/KV_QUANT.md`.

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::quant::KvQuant;
use crate::rotorquant::n_groups_for;
use crate::storage::{KvStorage, QuantIsoK3, QuantRotorK3};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

const MAX_SEQ: i32 = 1024;
const KV_H: i32 = 8;
const N_Q_HEADS: i32 = 32;
const HEAD_DIM: i32 = 128;

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Build an empty K-only cache for a ring-backed codec.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test helper covers exactly the two ring-backed K-only codecs it is called with; any other variant is a caller bug the unreachable! surfaces immediately"
)]
fn ring_backed_cache(quant: KvQuant) -> KvCache {
    let shape = vec![1, KV_H, 0, HEAD_DIM];
    let storage = match quant {
        KvQuant::IsoKOnly3 => KvStorage::IsoKOnly3 {
            k: Some(QuantIsoK3::from_cpu_blocks(Vec::new(), shape, MAX_SEQ)),
            max_seq: MAX_SEQ,
        },
        KvQuant::RotorKOnly3 => KvStorage::RotorKOnly3 {
            k: Some(QuantRotorK3::from_cpu_blocks(
                make_rotor_table(0, 0, n_groups_for(HEAD_DIM as usize)),
                None,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        },
        _ => unreachable!("ring_backed_cache covers the K-only ring codecs only"),
    };
    KvCache::from_storage(storage, quant, 0, 0)
}

/// The ring's own byte size, read from the store. `None` when no ring is live.
///
/// This is the accounting under test, not an independent oracle — see the
/// module docs. `ring_bytes_match_independent_geometry` is the oracle.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "only the ring-backed K-only variants carry a ring; every other storage variant correctly reports no live ring"
)]
fn live_ring_bytes(cache: &KvCache) -> Option<u64> {
    let (allocated, bytes) = match cache.storage() {
        KvStorage::IsoKOnly3 { k: Some(ks), .. } => (ks.gpu.is_allocated(), ks.gpu.byte_size()),
        KvStorage::RotorKOnly3 { k: Some(ks), .. } => (ks.gpu.is_allocated(), ks.gpu.byte_size()),
        _ => (false, 0),
    };
    allocated.then_some(bytes)
}

/// The ring's live capacity, read from its bookkeeping (not from its buffers).
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "only the ring-backed K-only variants carry a ring; every other storage variant has no capacity to report"
)]
fn live_ring_capacity(cache: &KvCache) -> Option<i32> {
    match cache.storage() {
        KvStorage::IsoKOnly3 { k: Some(ks), .. } if ks.gpu.is_allocated() => Some(ks.gpu.capacity),
        _ => None,
    }
}

/// Prefill `prefill` positions, then take one decode step — which is what
/// stands the ring up. Returns `resident_bytes` from just before that step.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn drive_to_ring(cache: &mut KvCache, prefill: i32) -> u64 {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    // Prefill goes through the legacy path: CPU blocks accumulate, no ring yet.
    let pf = (prefill * KV_H * HEAD_DIM) as usize;
    let k = f32_array(&lcg_data(pf, 1), &[1, KV_H, prefill, HEAD_DIM]);
    let v = f32_array(&lcg_data(pf, 2), &[1, KV_H, prefill, HEAD_DIM]);
    let q = f32_array(
        &lcg_data((prefill * N_Q_HEADS * HEAD_DIM) as usize, 3),
        &[1, N_Q_HEADS, prefill, HEAD_DIM],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    assert!(
        live_ring_bytes(cache).is_none(),
        "no ring should exist before the first decode dispatch"
    );
    let before = cache.resident_bytes();

    // First decode step seeds the ring from the accumulated CPU prefix.
    let one = (KV_H * HEAD_DIM) as usize;
    let k1 = f32_array(&lcg_data(one, 10), &[1, KV_H, 1, HEAD_DIM]);
    let v1 = f32_array(&lcg_data(one, 20), &[1, KV_H, 1, HEAD_DIM]);
    let q1 = f32_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 30),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    let out = cache
        .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
        .expect("decode update_and_sdpa");
    out.eval().expect("decode out eval");
    before
}

/// Prefill, then decode until the flash path stands the ring up.
///
/// Returns `(bytes_before_ring, bytes_after_ring, ring_bytes)`.
fn bytes_across_ring_allocation(quant: KvQuant, prefill: i32) -> (u64, u64, u64) {
    let mut cache = ring_backed_cache(quant);
    let before = drive_to_ring(&mut cache, prefill);
    let ring = live_ring_bytes(&cache).unwrap_or_else(|| {
        panic!("{quant}: decode must stand up the GPU ring — the flash path cannot run without it")
    });
    (before, cache.resident_bytes(), ring)
}

/// Standing up the ring must move the reported total by at least the ring's
/// own size.
///
/// `>=` not `==`: the same decode step also appends one position of K to the
/// CPU blocks and advances the bf16 V mirror's filled prefix, so the total
/// legitimately grows by a little more than the ring alone. What must never
/// happen again is a delta of zero.
fn assert_ring_is_counted(quant: KvQuant, prefill: i32) {
    let (before, after, ring) = bytes_across_ring_allocation(quant, prefill);
    assert!(
        ring > 0,
        "{quant}: a live ring must have non-zero size (prefill={prefill})"
    );
    let delta = after.saturating_sub(before);
    assert!(
        delta >= ring,
        "{quant} @ prefill={prefill}: resident_bytes grew by {delta} B but the ring alone \
         allocated {ring} B — the ring is not being counted (before={before}, after={after})"
    );
}

/// The ring's reported size matches its documented layout, derived independently.
///
/// This is the magnitude oracle the delta tests deliberately are not. It does
/// not call `byte_size`'s arithmetic: it rebuilds the total from the layout
/// `QuantKGpuRing` documents for its three buffers —
/// `codes: u32[cap * kv_h * n_groups]`, `scales: f32[cap * kv_h * n_groups]`,
/// `norms: f32[cap * kv_h]`, with iso's `n_groups = head_dim / 4` — and
/// compares. A dtype or element-count error in the accounting (the 4-vs-8 byte
/// class) fails here; the delta tests would sail through it.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn ring_bytes_match_independent_geometry() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (_, _, reported) = bytes_across_ring_allocation(KvQuant::IsoKOnly3, 256);

    // Rebuild the cache the same way to read its live capacity; the ring is
    // page-rounded, so the capacity is the one value that must come from it.
    let mut cache = ring_backed_cache(KvQuant::IsoKOnly3);
    drive_to_ring(&mut cache, 256);
    let cap = live_ring_capacity(&cache).expect("iso3 ring must be live after decode") as u64;

    let kv_h = KV_H as u64;
    let n_groups = HEAD_DIM as u64 / 4; // ISO_QUAT_BLOCK_SIZE
    let codes = cap * kv_h * n_groups * 4; // u32
    let scales = cap * kv_h * n_groups * 4; // f32
    let norms = cap * kv_h * 4; // f32
    let expected = codes + scales + norms;

    assert_eq!(
        reported, expected,
        "iso3 ring reports {reported} B but its documented layout at capacity={cap} \
         (kv_h={kv_h}, n_groups={n_groups}) is {expected} B \
         (codes={codes} + scales={scales} + norms={norms})"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn iso_k_only3_ring_is_counted_in_resident_bytes() {
    if skip_if_no_gpu_env() {
        return;
    }
    assert_ring_is_counted(KvQuant::IsoKOnly3, 256);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn rotor_k_only3_ring_is_counted_in_resident_bytes() {
    if skip_if_no_gpu_env() {
        return;
    }
    assert_ring_is_counted(KvQuant::RotorKOnly3, 256);
}

/// The ring stays counted as it grows with context.
///
/// A single context is not enough: the ring is allocated in pages and re-seeded
/// as the prefix grows, so an accounting that happened to be right at one size
/// can still be wrong at another. Sweeps both ring-backed codecs across two
/// contexts.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn ring_stays_counted_across_contexts() {
    if skip_if_no_gpu_env() {
        return;
    }
    for quant in [KvQuant::IsoKOnly3, KvQuant::RotorKOnly3] {
        for prefill in [128, 512] {
            assert_ring_is_counted(quant, prefill);
        }
    }
}
