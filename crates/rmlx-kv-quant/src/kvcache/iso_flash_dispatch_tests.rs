//! Production-path dispatch tests for the iso flash-decode kernel.
//!
//! The oracle tests in `iso_flash_decode_msl_tests.rs` prove the kernel is
//! numerically right when called **directly**, with hand-built arrays. These
//! prove the thing that actually matters in production: that
//! `KvCache::update_and_sdpa` **reaches** it on an iso K-only cache, instead of
//! silently falling through to the O(seq) CPU dequant path the kernel exists to
//! remove — and that the ring bookkeeping around it (seed / grow / feed / skip)
//! holds. That growth-and-seed logic is exactly what carried the rotor ring's
//! frozen-`max_seq` defect, and a direct-call oracle cannot see any of it.
//!
//! Sibling of `rotor_flash_dispatch_tests.rs`.
//!
//! # No env dependence
//!
//! Iso has no QJL sideband and no CLI/env gate, so unlike the rotor siblings
//! these tests need no toggle-pinning: the path is on whenever it is applicable.

use super::KvCache;
use crate::iso_flash_decode_msl::iso_flash_decode_dispatch_count;
use crate::quant::KvQuant;
use crate::storage::{KvStorage, QuantIsoK3, QuantIsoK4};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

const MAX_SEQ: i32 = 512;

/// What one `run_decode_steps` call observed.
struct DecodeProbe {
    /// Dispatch-count delta across the decode steps.
    ///
    /// The counter is process-global, so a concurrent test can only ever
    /// *inflate* this — `>=` assertions on it are race-free, `== 0` ones are
    /// not. Use `gpu_ring_live` for the negative case.
    delta: u64,
    /// Whether the iso store's GPU ring is live at the end of the run.
    ///
    /// Local to this cache, so unlike `delta` it is immune to other tests. The
    /// flash path cannot run without the ring, so `!gpu_ring_live` is a
    /// race-free proof that decode stayed on the CPU dequant path.
    gpu_ring_live: bool,
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Build an empty iso K-only cache.
fn iso_cache(quant: KvQuant, kv_h: i32, head_dim: i32) -> KvCache {
    iso_cache_b(quant, 1, kv_h, head_dim)
}

/// [`iso_cache`] with an explicit batch dimension.
fn iso_cache_b(quant: KvQuant, b: i32, kv_h: i32, head_dim: i32) -> KvCache {
    let shape = vec![b, kv_h, 0, head_dim];
    let storage = if quant == KvQuant::IsoKOnly4 {
        KvStorage::IsoKOnly4 {
            k: Some(QuantIsoK4::from_cpu_blocks(Vec::new(), shape, MAX_SEQ)),
            max_seq: MAX_SEQ,
        }
    } else {
        KvStorage::IsoKOnly3 {
            k: Some(QuantIsoK3::from_cpu_blocks(Vec::new(), shape, MAX_SEQ)),
            max_seq: MAX_SEQ,
        }
    };
    KvCache::from_storage(storage, quant, 0, 0, DispatchPolicy::default())
}

/// Whether the cache's iso store currently holds a live GPU ring.
fn ring_live(cache: &KvCache) -> bool {
    if let KvStorage::IsoKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else if let KvStorage::IsoKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else {
        false
    }
}

/// Drive one prefill chunk + 4 decode steps through the production
/// `update_and_sdpa` entry point.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn run_decode_steps(
    quant: KvQuant,
    kv_h: i32,
    n_q_heads: i32,
    head_dim: i32,
    prefill: i32,
) -> DecodeProbe {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut cache = iso_cache(quant, kv_h, head_dim);

    // Prefill chunk (q_seq > 1): the flash path is decode-only, so this goes
    // through the legacy path and leaves an accumulated prefix behind —
    // exactly the state the first decode step must seed the GPU ring from.
    let pf = (prefill * kv_h * head_dim) as usize;
    let k = f32_array(&lcg_data(pf, 1), &[1, kv_h, prefill, head_dim]);
    let v = f32_array(&lcg_data(pf, 2), &[1, kv_h, prefill, head_dim]);
    let q = f32_array(
        &lcg_data((prefill * n_q_heads * head_dim) as usize, 3),
        &[1, n_q_heads, prefill, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    let before = iso_flash_decode_dispatch_count();
    for step in 0..4_u64 {
        let one = (kv_h * head_dim) as usize;
        let k1 = f32_array(&lcg_data(one, 10 + step), &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(&lcg_data(one, 20 + step), &[1, kv_h, 1, head_dim]);
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 30 + step),
            &[1, n_q_heads, 1, head_dim],
        );
        let out = cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa");
        out.eval().expect("decode out eval");
        assert_eq!(
            out.shape(),
            vec![1, n_q_heads, 1, head_dim],
            "decode output shape"
        );
    }
    let delta = iso_flash_decode_dispatch_count() - before;
    let gpu_ring_live = ring_live(&cache);
    DecodeProbe {
        delta,
        gpu_ring_live,
    }
}

// ── The kernel is reached from the production entry point ─────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso_k_only_3_decode_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let probe = run_decode_steps(KvQuant::IsoKOnly3, 2, 8, 128, 24);
    assert!(
        probe.delta >= 4,
        "iso3 decode did not dispatch the flash kernel (delta={}) — it fell through \
         to the CPU dequant path",
        probe.delta
    );
    assert!(probe.gpu_ring_live, "iso3: the GPU ring must be live");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso_k_only_4_decode_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let probe = run_decode_steps(KvQuant::IsoKOnly4, 2, 8, 128, 24);
    assert!(
        probe.delta >= 4,
        "iso4 decode did not dispatch the flash kernel (delta={})",
        probe.delta
    );
    assert!(probe.gpu_ring_live, "iso4: the GPU ring must be live");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso_k_only_decode_dispatches_across_a_tile_boundary() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Prefill 62 + 4 decode steps crosses TILE_SIZE (64), so the ring grows and
    // the P2 log-sum-exp merge runs over more than one tile.
    let probe = run_decode_steps(KvQuant::IsoKOnly3, 2, 8, 128, 62);
    assert!(
        probe.delta >= 4,
        "tile-boundary decode did not dispatch (delta={})",
        probe.delta
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso_k_only_decode_dispatches_at_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Gemma3-class shape. The kernel keys off shape, not arch.
    let probe = run_decode_steps(KvQuant::IsoKOnly3, 4, 8, 256, 24);
    assert!(
        probe.delta >= 4,
        "head_dim=256 decode did not dispatch (delta={})",
        probe.delta
    );
}

// ── Ring feed: b > 1 skips rather than errors ─────────────────────────────────

/// `b > 1` must skip the ring feed, not attempt (and fail) it.
///
/// `QuantKGpuRing`'s per-step stride is `kv_h * n_groups` and does not interleave
/// batch, so a batched chunk cannot be laid into it: the encode arrays carry
/// `b * kv_h * new_seq * n_groups` entries against a `new_seq * kv_h * n_groups`
/// span. Attempting it returns `Err` and kills the request, where the CPU blocks
/// (which handle `b > 1`) would have served it fine.
fn batched_ring_feed_is_skipped(quant: KvQuant, bits_label: &str) {
    let device = Device::Gpu;
    let (b, kv_h, head_dim, new_seq) = (2_i32, 2_i32, 128_i32, 4_i32);
    let mut cache = iso_cache_b(quant, b, kv_h, head_dim);
    let shape = [b, kv_h, new_seq, head_dim];
    let k = f32_array(
        &lcg_data((b * kv_h * new_seq * head_dim) as usize, 7),
        &shape,
    );

    let res = if quant == KvQuant::IsoKOnly4 {
        super::update::iso4_k_only_gpu_append(&mut cache, &k, &shape, device)
    } else {
        super::update::iso3_k_only_gpu_append(&mut cache, &k, &shape, device)
    };
    res.unwrap_or_else(|e| {
        panic!("{bits_label}: batched GPU append must not error, got: {e}");
    });

    assert!(
        !ring_live(&cache),
        "{bits_label}: a b>1 cache must not carry a ring — the stride cannot represent it"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso3_batched_gpu_append_skips_the_ring_instead_of_erroring() {
    if skip_if_no_gpu_env() {
        return;
    }
    batched_ring_feed_is_skipped(KvQuant::IsoKOnly3, "iso3");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso4_batched_gpu_append_skips_the_ring_instead_of_erroring() {
    if skip_if_no_gpu_env() {
        return;
    }
    batched_ring_feed_is_skipped(KvQuant::IsoKOnly4, "iso4");
}

// ── Ring invariant: a CPU append must drop a live ring ────────────────────────

/// A CPU `append` after a GPU append must **drop** the ring, not leave it short.
///
/// The invariant (see `RingFeed`): the ring either tracks `blocks` exactly, or
/// it does not exist. A CPU append grows `blocks` and `shape[2]` without
/// touching the ring, so leaving the ring live is the dangerous state — the next
/// `gpu_append` takes `prev_seq` from the longer `shape` and writes past the
/// ring's filled region, leaving `[ring_filled, prev_seq)` zeroed. The kernel
/// then attends a zero-filled hole with **no error**.
///
/// Reachable in production: `update_iso_k_only_{3,4}` / `update_iso{3,4}_sym`
/// take `ks.append(..)` on the `device != Gpu` branch while the fused decode
/// entry feeds with `RingFeed::Maintain`, so a Gpu→Cpu→Gpu sequence on one cache
/// hits it. The field is `pub`, so the store cannot rely on its callers either.
fn cpu_append_drops_a_live_ring(quant: KvQuant, bits_label: &str) {
    let device = Device::Gpu;
    let (kv_h, head_dim, new_seq) = (2_i32, 128_i32, 4_i32);
    let mut cache = iso_cache(quant, kv_h, head_dim);
    let shape = [1_i32, kv_h, new_seq, head_dim];
    let n = (kv_h * new_seq * head_dim) as usize;
    let k = f32_array(&lcg_data(n, 11), &shape);

    // 1. GPU append -> ring live.
    if quant == KvQuant::IsoKOnly4 {
        super::update::iso4_k_only_gpu_append(&mut cache, &k, &shape, device)
    } else {
        super::update::iso3_k_only_gpu_append(&mut cache, &k, &shape, device)
    }
    .unwrap_or_else(|e| panic!("{bits_label}: gpu_append: {e}"));
    assert!(
        ring_live(&cache),
        "{bits_label}: precondition — the GPU append must leave a live ring, else this \
         test proves nothing"
    );

    // 2. CPU append on the same store.
    let cpu_data = lcg_data(n, 12);
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "the wildcard arm panics — it never selects another codec's store, and \
                  `iso_cache` built this one, so reaching it is a broken test invariant \
                  rather than a variant that needs handling"
    )]
    match &mut cache.storage {
        KvStorage::IsoKOnly3 { k: Some(ks), .. } => ks.append(&cpu_data, &shape),
        KvStorage::IsoKOnly4 { k: Some(ks), .. } => ks.append(&cpu_data, &shape),
        _ => panic!("{bits_label}: store vanished"),
    }
    .unwrap_or_else(|e| panic!("{bits_label}: cpu append: {e}"));

    // 3. The ring must be gone — it no longer tracks `blocks`.
    assert!(
        !ring_live(&cache),
        "{bits_label}: a CPU append left a live GPU ring behind. `blocks` and `shape[2]` \
         grew but the ring did not, so the next gpu_append writes past its filled region \
         and the kernel silently attends a zero-filled hole."
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso3_cpu_append_drops_a_live_gpu_ring() {
    if skip_if_no_gpu_env() {
        return;
    }
    cpu_append_drops_a_live_ring(KvQuant::IsoKOnly3, "iso3");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test iso_flash_dispatch -- --ignored --test-threads=1"]
fn iso4_cpu_append_drops_a_live_gpu_ring() {
    if skip_if_no_gpu_env() {
        return;
    }
    cpu_append_drops_a_live_ring(KvQuant::IsoKOnly4, "iso4");
}
