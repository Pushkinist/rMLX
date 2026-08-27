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
//! These pin the ring into the total: once it is live it is the K store's whole
//! payload (the CPU blocks are dropped in the same step), so the store's own
//! byte total must equal it, across all four ring-backed K-only codecs (both
//! bit widths of both families) and two contexts.
//! The bug that was shipped — a total blind to the ring — reads 0 here.
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
use crate::storage::{
    KvStorage, QuantIsoK3, QuantIsoK4, QuantRotorK3, QuantRotorK4, KV_SIDEBAND_DTYPE,
};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

const MAX_SEQ: i32 = 1024;
const KV_H: i32 = 8;
const N_Q_HEADS: i32 = 32;
const HEAD_DIM: i32 = 128;

/// A `bf16` input array.
///
/// bf16 and not f32 because the mirror buffers this file measures take their
/// dtype from the stream that fills them (`stream_dtype`): an f32 K/V feed
/// produces f32 mirrors, which are twice the bytes production allocates and
/// would make every mirror term here describe a cache no request ever holds.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn bf16_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32)
        .expect("bf16_array: build f32")
        .astype(Dtype::Bf16, Device::Gpu)
        .expect("bf16_array: cast")
}

/// Build an empty K-only cache for a ring-backed codec.
///
/// All four members of the two ring-backed K-only families are covered. The
/// bit-width siblings share a store layout and a fused decode arm, so a helper
/// that only knew the 3-bit ones would leave the 4-bit halves of both families
/// undriven — the exact "one member of a sibling pair was missed" shape these
/// tests exist to catch.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test helper covers exactly the four ring-backed K-only codecs it is called with; any other variant is a caller bug the unreachable! surfaces immediately"
)]
fn ring_backed_cache(quant: KvQuant) -> KvCache {
    let shape = vec![1, KV_H, 0, HEAD_DIM];
    let rotor_table = || make_rotor_table(0, 0, n_groups_for(HEAD_DIM as usize));
    let storage = match quant {
        KvQuant::IsoKOnly3 => KvStorage::IsoKOnly3 {
            k: Some(QuantIsoK3::from_cpu_blocks(Vec::new(), shape, MAX_SEQ)),
            max_seq: MAX_SEQ,
        },
        KvQuant::IsoKOnly4 => KvStorage::IsoKOnly4 {
            k: Some(QuantIsoK4::from_cpu_blocks(Vec::new(), shape, MAX_SEQ)),
            max_seq: MAX_SEQ,
        },
        KvQuant::RotorKOnly3 => KvStorage::RotorKOnly3 {
            k: Some(QuantRotorK3::from_cpu_blocks(
                rotor_table(),
                None,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        },
        KvQuant::RotorKOnly4 => KvStorage::RotorKOnly4 {
            k: Some(QuantRotorK4::from_cpu_blocks(
                rotor_table(),
                None,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        },
        _ => unreachable!("ring_backed_cache covers the K-only ring codecs only"),
    };
    KvCache::from_storage(storage, quant, 0, 0, DispatchPolicy::default())
}

/// Every ring-backed K-only codec, both bit widths of both families.
const RING_BACKED_K_ONLY: [KvQuant; 4] = [
    KvQuant::IsoKOnly3,
    KvQuant::IsoKOnly4,
    KvQuant::RotorKOnly3,
    KvQuant::RotorKOnly4,
];

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
        KvStorage::IsoKOnly4 { k: Some(ks), .. } => (ks.gpu.is_allocated(), ks.gpu.byte_size()),
        KvStorage::RotorKOnly3 { k: Some(ks), .. } => (ks.gpu.is_allocated(), ks.gpu.byte_size()),
        KvStorage::RotorKOnly4 { k: Some(ks), .. } => (ks.gpu.is_allocated(), ks.gpu.byte_size()),
        _ => (false, 0),
    };
    allocated.then_some(bytes)
}

/// Tokens still held by the store's CPU blocks. `None` when the store is absent.
///
/// Once the ring is live it is the sole resident copy of the packed K prefix, so
/// this must be `Some(0)` for every ring-backed K-only codec. A non-zero count
/// there is the prefill prefix retained a second time, on top of the ring that
/// already holds it.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "only the ring-backed K-only variants carry CPU blocks alongside a ring; every other storage variant has none to report"
)]
fn cpu_block_tokens(cache: &KvCache) -> Option<usize> {
    match cache.storage() {
        KvStorage::IsoKOnly3 { k: Some(ks), .. } => {
            Some(ks.blocks.iter().map(|b| b.n_tokens).sum())
        }
        KvStorage::IsoKOnly4 { k: Some(ks), .. } => {
            Some(ks.blocks.iter().map(|b| b.n_tokens).sum())
        }
        KvStorage::RotorKOnly3 { k: Some(ks), .. } => {
            Some(ks.blocks.iter().map(|b| b.n_tokens).sum())
        }
        KvStorage::RotorKOnly4 { k: Some(ks), .. } => {
            Some(ks.blocks.iter().map(|b| b.n_tokens).sum())
        }
        _ => None,
    }
}

/// Bytes the K store reports for itself (CPU blocks + GPU ring + sidebands).
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "only the ring-backed K-only variants are driven here; any other storage variant holds no K store to size"
)]
fn k_store_bytes(cache: &KvCache) -> u64 {
    match cache.storage() {
        KvStorage::IsoKOnly3 { k: Some(ks), .. } => ks.byte_size(),
        KvStorage::IsoKOnly4 { k: Some(ks), .. } => ks.byte_size(),
        KvStorage::RotorKOnly3 { k: Some(ks), .. } => ks.byte_size(),
        KvStorage::RotorKOnly4 { k: Some(ks), .. } => ks.byte_size(),
        _ => 0,
    }
}

/// Bytes the K store holds that are **not** per-token: rotor's static
/// per-(layer, head) rotation table and its optional QJL projection, which are
/// generated once and amortise over every token in the layer. Iso's rotation is
/// a compile-time constant (`FIXED_QUAT`), so it stores none.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "only the rotor store carries a static rotation sideband; every other variant here has none"
)]
fn k_store_static_bytes(cache: &KvCache) -> u64 {
    let f32_bytes = |n: usize| (n * size_of::<f32>()) as u64;
    match cache.storage() {
        KvStorage::RotorKOnly3 { k: Some(ks), .. } => {
            f32_bytes(ks.rotors.len()) + ks.qjl_s_matrix.as_ref().map_or(0, |m| f32_bytes(m.len()))
        }
        KvStorage::RotorKOnly4 { k: Some(ks), .. } => {
            f32_bytes(ks.rotors.len()) + ks.qjl_s_matrix.as_ref().map_or(0, |m| f32_bytes(m.len()))
        }
        _ => 0,
    }
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
/// stands the ring up.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn drive_to_ring(cache: &mut KvCache, prefill: i32) {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

    // `enter_prefill()` is load-bearing here, not cosmetic: without it
    // `in_prefill` stays `false` (the `from_storage` / `with_quant_max_seq`
    // default), so the "prefill" call below does not take the raw bf16
    // accumulator path (`update_prefill_raw`) — it falls through
    // `update_and_sdpa`'s legacy arm straight into the per-codec decode-style
    // updater (`update_rotor_k_only_3`), which — for RotorKOnly3, once
    // `--rotor-qjl` defaults off — GPU-encodes and stands the ring up right
    // there, even for a multi-token chunk. That is a real production path too
    // (e.g. an SSD-hydrated cache fed a multi-token catch-up chunk), but it is
    // not what this helper means by "prefill": calling `enter_prefill()`
    // routes the chunk through the true raw-accumulator prefill path (ring
    // stays dormant until the first single-token decode step), matching every
    // other codec's `drive_to_ring` behaviour and restoring the precondition
    // this test actually wants to check.
    cache.enter_prefill();

    // Prefill goes through the legacy path: CPU blocks accumulate, no ring yet.
    let pf = (prefill * KV_H * HEAD_DIM) as usize;
    let k = bf16_array(&lcg_data(pf, 1), &[1, KV_H, prefill, HEAD_DIM]);
    let v = bf16_array(&lcg_data(pf, 2), &[1, KV_H, prefill, HEAD_DIM]);
    let q = bf16_array(
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

    // First decode step seeds the ring from the accumulated CPU prefix.
    let one = (KV_H * HEAD_DIM) as usize;
    let k1 = bf16_array(&lcg_data(one, 10), &[1, KV_H, 1, HEAD_DIM]);
    let v1 = bf16_array(&lcg_data(one, 20), &[1, KV_H, 1, HEAD_DIM]);
    let q1 = bf16_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 30),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    let out = cache
        .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
        .expect("decode update_and_sdpa");
    out.eval().expect("decode out eval");
}

/// Prefill, then decode until the flash path stands the ring up.
///
/// Returns the driven cache and its live ring size.
fn cache_with_live_ring(quant: KvQuant, prefill: i32) -> (KvCache, u64) {
    let mut cache = ring_backed_cache(quant);
    drive_to_ring(&mut cache, prefill);
    let ring = live_ring_bytes(&cache).unwrap_or_else(|| {
        panic!("{quant}: decode must stand up the GPU ring — the flash path cannot run without it")
    });
    (cache, ring)
}

/// The live ring is the K store's whole per-token payload, and it reaches the
/// cache total.
///
/// Once the ring is live the store drops its CPU blocks, so everything the
/// store holds is the ring plus its static rotation sideband. That makes this
/// the sharpest form of the check this file exists for: a total blind to the
/// ring reports the sideband alone here (0 B for iso), not a number that is
/// merely too small. (An earlier form compared the total across the allocating
/// decode step; that cannot work, because the same step also releases the
/// blocks — the delta is a difference of two large terms, not the ring.)
fn assert_ring_is_counted(quant: KvQuant, prefill: i32) {
    let (cache, ring) = cache_with_live_ring(quant, prefill);
    assert!(
        ring > 0,
        "{quant}: a live ring must have non-zero size (prefill={prefill})"
    );
    let store = k_store_bytes(&cache);
    let expected = ring + k_store_static_bytes(&cache);
    assert_eq!(
        store, expected,
        "{quant} @ prefill={prefill}: the K store reports {store} B but its live ring plus \
         static rotation sideband is {expected} B — with the CPU blocks dropped the two must \
         be the same number"
    );
    // The slack over the ring is pinned, not left open. `total >= ring` alone is
    // satisfiable by a total that has stopped counting the bf16 V mirror these
    // K-only codecs decode V from — it would keep passing while silently
    // measuring less than it claims to. The mirror's size is derived here from
    // geometry (`kv_h * filled * head_dim` bf16 elements), independently of the
    // accounting under test, and `filled` is the prefill chunk plus the one
    // decode step `drive_to_ring` takes. A lower bound on purpose: a codec that
    // also keeps a K seed, or a mirror at a wider dtype, only makes `total`
    // larger.
    let total = cache.resident_bytes();
    let filled = (prefill + 1) as u64;
    let v_mirror = (KV_H as u64) * filled * (HEAD_DIM as u64) * 2;
    let floor = ring + v_mirror;
    assert!(
        total >= floor,
        "{quant} @ prefill={prefill}: resident_bytes is {total} B, below the {floor} B it must \
         hold ({ring} B ring + {v_mirror} B bf16 V mirror over {filled} filled positions) — \
         either the ring is not reaching the cache-level total or the V mirror stopped counting"
    );
}

/// The ring's reported size matches its documented layout, derived independently.
///
/// This is the magnitude oracle the delta tests deliberately are not. It does
/// not call `byte_size`'s arithmetic: it rebuilds the total from the layout
/// `QuantKGpuRing` documents for its three buffers —
/// `codes: u32[cap * kv_h * n_groups]` and the two sideband planes
/// `scales[cap * kv_h * n_groups]` / `norms[cap * kv_h]` at
/// `KV_SIDEBAND_DTYPE`, with iso's `n_groups = head_dim / 4` — and compares. A
/// dtype or element-count error in the accounting (the 4-vs-8 byte class) fails
/// here; the delta tests would sail through it.
///
/// The two widths are read from `Dtype::itemsize` rather than written as
/// literals, but they are read from *different* sources than `byte_size` uses:
/// this side names `Dtype::U32` and `KV_SIDEBAND_DTYPE` explicitly, while
/// `byte_size` reads whatever dtype each allocated buffer actually carries. A
/// plane allocated at the wrong dtype still fails here.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn ring_bytes_match_independent_geometry() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (cache, reported) = cache_with_live_ring(KvQuant::IsoKOnly3, 256);
    // The ring is page-rounded, so the capacity is the one value that must come
    // from it rather than from the prefill length.
    let cap = live_ring_capacity(&cache).expect("iso3 ring must be live after decode") as u64;

    let kv_h = KV_H as u64;
    let n_groups = HEAD_DIM as u64 / 4; // ISO_QUAT_BLOCK_SIZE
    let code_w = Dtype::U32.itemsize() as u64;
    let side_w = KV_SIDEBAND_DTYPE.itemsize() as u64;
    let codes = cap * kv_h * n_groups * code_w;
    let scales = cap * kv_h * n_groups * side_w;
    let norms = cap * kv_h * side_w;
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

/// A live ring makes the CPU blocks redundant — every ring-backed K-only codec
/// must drop them.
///
/// The ring and the blocks hold the same packed prefix in the same layout, so a
/// store that keeps both pays for the prefix twice and reports a KV total well
/// above what its own format costs. Swept over **all four** ring-backed K-only
/// codecs — both bit widths of both families — and two contexts: the drop is a
/// property of "the ring is the sole store", not of one codec at one size.
///
/// Sweeping the bit-width siblings is the point, not thoroughness for its own
/// sake. The defect this guards was one member of a sibling pair missing the
/// drop call while the other had it; each width has its own append function and
/// its own call site, so covering only the 3-bit ones would leave the 4-bit
/// halves free to repeat it.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn ring_backed_k_only_codecs_drop_their_cpu_blocks() {
    if skip_if_no_gpu_env() {
        return;
    }
    for quant in RING_BACKED_K_ONLY {
        for prefill in [128, 512] {
            let (cache, _ring) = cache_with_live_ring(quant, prefill);
            assert_eq!(
                cpu_block_tokens(&cache),
                Some(0),
                "{quant} @ prefill={prefill}: the ring is live, so the CPU blocks are a \
                 second copy of the same packed prefix and must have been dropped"
            );
        }
    }
}

/// The ring stays counted as it grows with context.
///
/// A single context is not enough: the ring is allocated in pages and re-seeded
/// as the prefix grows, so an accounting that happened to be right at one size
/// can still be wrong at another. Sweeps all four ring-backed K-only codecs
/// across two contexts.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn ring_stays_counted_across_contexts() {
    if skip_if_no_gpu_env() {
        return;
    }
    for quant in RING_BACKED_K_ONLY {
        for prefill in [128, 512] {
            assert_ring_is_counted(quant, prefill);
        }
    }
}

/// The **estimator's** byte model, checked against a live cache rather than
/// against a store's own encoder.
///
/// `KvQuant::estimated_resident_bytes_per_layer` is what the resolve-time
/// net-benefit advisory is computed from, and every other gate on it —
/// `every_codec_byte_model_matches_the_store_it_writes`, the stored-rate
/// families in `kv_rate_tests` — checks it against a store built by hand from
/// an encoder's output. None of them drives a decode. That leaves one seam
/// uncovered: the estimator sizes the ring-backed families from
/// `SideStore::IsoRing` / `SideStore::Rotor`, which is only the right store
/// **because** the production append drops the CPU blocks once the ring is live
/// (`drop_blocks_when_ring_live_*`). If that ever regained a route where the
/// blocks survive, the iso model would be wrong by the block form's whole
/// factor and every hand-built gate would still be green.
///
/// So this one drives a real prefill and a real fused decode step, then
/// compares the estimate against `KvCache::resident_bytes`.
///
/// Two terms have to be named for the comparison to be exact, and both are
/// documented properties of the estimate rather than fudge:
///
/// * **Page rounding.** The ring allocates in whole `KV_PAGE_SIZE` pages and
///   the estimate models none, so the prefill is chosen to land the filled
///   length exactly on a page boundary. `live_ring_capacity` is asserted equal
///   to it, so a change in the growth policy fails here instead of being
///   absorbed.
/// * **Rotor's static rotation table.** Per-(layer, head), not per-token; the
///   estimate omits it on purpose ("estimate, not census"), so it is added
///   back from the store. It is zero for iso, whose rotation is a compile-time
///   constant.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn estimator_matches_a_live_ring_backed_cache() {
    if skip_if_no_gpu_env() {
        return;
    }
    // One decode step follows the prefill, so this lands `filled` on 256 — one
    // whole KV_PAGE_SIZE.
    const PREFILL: i32 = 255;
    let seq = u64::try_from(PREFILL + 1).unwrap_or(0);

    for quant in RING_BACKED_K_ONLY {
        let (cache, ring) = cache_with_live_ring(quant, PREFILL);
        assert!(ring > 0, "{quant}: the decode step must stand up the ring");

        let estimate = quant.estimated_resident_bytes_per_layer(seq, HEAD_DIM as u64, KV_H as u64);
        let expected = estimate + k_store_static_bytes(&cache);
        let actual = cache.resident_bytes();
        assert_eq!(
            actual,
            expected,
            "{quant} @ seq={seq}: the live cache holds {actual} B but the estimator models \
             {estimate} B (+{} B of static rotation table). The estimate sizes the K side \
             from the ring alone — if the CPU blocks survived the decode step, or a bf16 \
             seed came back, this is where it shows",
            k_store_static_bytes(&cache)
        );
    }
}

/// The estimate is the *ring* form, not the CPU-block form — and the gap is
/// large enough that nothing else in this file would notice the difference.
///
/// Companion to `estimator_matches_a_live_ring_backed_cache`: that one asserts
/// the estimate is right, this one asserts it is right for a reason, by pinning
/// what it would have been had the blocks survived. Without it, an estimate
/// that silently switched to the block form could still be made to pass by
/// moving the model and the gate together.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test resident_ring -- --ignored --test-threads=1"]
fn the_live_cache_estimate_is_the_ring_form_not_the_block_form() {
    if skip_if_no_gpu_env() {
        return;
    }
    const PREFILL: i32 = 255;
    let quant = KvQuant::IsoKOnly3;
    let (cache, ring) = cache_with_live_ring(quant, PREFILL);
    let cap = u64::try_from(
        live_ring_capacity(&cache).expect("iso3 ring must be live after the decode step"),
    )
    .unwrap_or(0);
    assert_eq!(
        cap,
        u64::try_from(PREFILL + 1).unwrap_or(0),
        "prefill is chosen so the ring's page-rounded capacity is the filled length"
    );

    // What the same prefix costs as CPU `IsoBlocks`: the ring's payload with
    // both sideband planes at f32, plus a replicated FIXED_QUAT per group.
    let kv_h = KV_H as u64;
    let n_groups = HEAD_DIM as u64 / 4;
    let groups = cap * kv_h * n_groups;
    let blocks = groups * (4 + 4 + 4 * 4) + cap * kv_h * 4;
    assert!(
        blocks > 3 * ring,
        "the block form ({blocks} B) must be several times the ring ({ring} B) — if it were \
         not, sizing the estimate from the wrong one would be undetectable"
    );
}
