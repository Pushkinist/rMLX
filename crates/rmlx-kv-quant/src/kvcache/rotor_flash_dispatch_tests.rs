//! Production-path dispatch tests for the rotor flash-decode kernel.
//!
//! The oracle tests in `rotor_flash_decode_msl_tests.rs` prove the kernel is
//! numerically right when called directly. These prove the thing that actually
//! matters in production: that `KvCache::update_and_sdpa` **reaches** it on a
//! rotor K-only cache, instead of silently falling through to the O(seq) CPU
//! dequant path the kernel exists to remove.
//!
//! # No env dependence
//!
//! These tests never touch `RMLX_ROTOR_QJL`. The codec fixes its QJL decision
//! on the first append (`rotors.is_empty()` lazy-init) and never revisits it, so
//! seeding a store with a pre-built rotor table pins QJL for that store's
//! lifetime — deterministically, whatever any other test in the binary does to
//! the process-global env var.

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::quant::KvQuant;
use crate::rotor_flash_decode_msl::rotor_flash_decode_dispatch_count;
use crate::rotorquant::{make_qjl_projection, n_groups_for};
use crate::storage::{KvStorage, QuantRotorK3, QuantRotorK4};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
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
    /// Whether the rotor store's GPU ring is live at the end of the run.
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

/// Build a rotor K-only cache whose QJL decision is pinned to `use_qjl`,
/// independent of the process-global toggle.
///
/// Pre-seeding `rotors` is what pins it: both the CPU and GPU append paths only
/// touch `qjl_s_matrix` inside their `rotors.is_empty()` lazy-init branch, so a
/// store handed a non-empty rotor table keeps whatever `qjl_s_matrix` it was
/// built with.
fn seeded_cache(quant: KvQuant, kv_h: i32, head_dim: i32, use_qjl: bool) -> KvCache {
    seeded_cache_b(quant, 1, kv_h, head_dim, use_qjl)
}

/// [`seeded_cache`] with an explicit batch dimension.
fn seeded_cache_b(quant: KvQuant, b: i32, kv_h: i32, head_dim: i32, use_qjl: bool) -> KvCache {
    let n_groups = n_groups_for(head_dim as usize);
    let rotors = make_rotor_table(0, 0, n_groups);
    let qjl = use_qjl.then(|| make_qjl_projection(head_dim as usize));
    let shape = vec![b, kv_h, 0, head_dim];

    let storage = if quant == KvQuant::RotorKOnly4 {
        KvStorage::RotorKOnly4 {
            k: Some(QuantRotorK4::from_cpu_blocks(
                rotors,
                qjl,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        }
    } else {
        KvStorage::RotorKOnly3 {
            k: Some(QuantRotorK3::from_cpu_blocks(
                rotors,
                qjl,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        }
    };
    KvCache::from_storage(storage, quant, 0, 0)
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
    use_qjl: bool,
) -> DecodeProbe {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut cache = seeded_cache(quant, kv_h, head_dim, use_qjl);

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

    let before = rotor_flash_decode_dispatch_count();
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
    let delta = rotor_flash_decode_dispatch_count() - before;
    let gpu_ring_live = if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else {
        false
    };
    DecodeProbe {
        delta,
        gpu_ring_live,
    }
}

#[test]
fn rotor_k_only_3_decode_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 2, 8, 128, 24, false);
    assert!(
        probe.delta >= 4,
        "update_and_sdpa did not reach the rotor3 flash-decode kernel on all 4 \
         decode steps (delta={}) — the codec is still CPU-dequanting the whole \
         K prefix every step",
        probe.delta
    );
    assert!(probe.gpu_ring_live, "rotor3 GPU ring should be live");
}

#[test]
fn rotor_k_only_4_decode_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let probe = run_decode_steps(KvQuant::RotorKOnly4, 2, 8, 128, 24, false);
    assert!(
        probe.delta >= 4,
        "update_and_sdpa did not reach the rotor4 flash-decode kernel (delta={})",
        probe.delta
    );
    assert!(probe.gpu_ring_live, "rotor4 GPU ring should be live");
}

#[test]
fn rotor_k_only_decode_dispatches_across_a_tile_boundary() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Prefill past TILE_SIZE so the decode steps span multiple pass-1 tiles and
    // exercise the pass-2 merge on the production path.
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 1, 4, 128, 100, false);
    assert!(
        probe.delta >= 4,
        "multi-tile decode did not dispatch (delta={})",
        probe.delta
    );
}

#[test]
fn rotor_k_only_decode_dispatches_at_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Gemma3-class shape. The kernel keys off shape, not arch.
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 4, 8, 256, 24, false);
    assert!(
        probe.delta >= 4,
        "head_dim=256 decode did not dispatch (delta={})",
        probe.delta
    );
}

/// `b > 1` must skip the ring feed, not attempt (and fail) it.
///
/// `QuantKGpuRing`'s per-step stride is `kv_h * n_groups` and does not interleave
/// batch, so a batched chunk cannot be laid into it: the encode arrays carry
/// `b * kv_h * new_seq * n_groups` entries against a `new_seq * kv_h * n_groups`
/// span. Attempting it returns `Err` and kills the request, where the CPU blocks
/// (which handle `b > 1`) would have served it fine.
///
/// Drives the ring-maintaining entry point **directly** rather than through
/// `update_and_sdpa`: the legacy `update_rotor_k_only_*` gates its GPU encode on
/// the global `rotor_qjl_enabled()` env (default on), so a store-seeded
/// end-to-end test would silently take the CPU append and never reach this
/// guard — i.e. pass vacuously.
fn batched_ring_feed_is_skipped(quant: KvQuant, bits_label: &str) {
    let device = Device::Gpu;
    let (b, kv_h, head_dim, new_seq) = (2_i32, 2_i32, 128_i32, 4_i32);
    let mut cache = seeded_cache_b(quant, b, kv_h, head_dim, false);
    let shape = [b, kv_h, new_seq, head_dim];
    let k = f32_array(
        &lcg_data((b * kv_h * new_seq * head_dim) as usize, 7),
        &shape,
    );

    let res = if quant == KvQuant::RotorKOnly4 {
        super::update::rotor4_k_only_gpu_append(&mut cache, &k, &shape, device)
    } else {
        super::update::rotor3_k_only_gpu_append(&mut cache, &k, &shape, device)
    };
    res.unwrap_or_else(|e| {
        panic!("{bits_label}: batched GPU append must not error, got: {e}");
    });

    let ring_live = if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else {
        panic!("{bits_label}: store vanished");
    };
    assert!(
        !ring_live,
        "{bits_label}: a b>1 cache must not carry a ring — the stride cannot represent it"
    );
}

#[test]
fn rotor3_batched_gpu_append_skips_the_ring_instead_of_erroring() {
    if skip_if_no_gpu_env() {
        return;
    }
    batched_ring_feed_is_skipped(KvQuant::RotorKOnly3, "rotor3");
}

#[test]
fn rotor4_batched_gpu_append_skips_the_ring_instead_of_erroring() {
    if skip_if_no_gpu_env() {
        return;
    }
    batched_ring_feed_is_skipped(KvQuant::RotorKOnly4, "rotor4");
}

#[test]
fn rotor_k_only_decode_stays_cpu_when_store_carries_qjl() {
    if skip_if_no_gpu_env() {
        return;
    }
    // The kernel cannot reproduce the QJL residual, so a QJL-carrying store must
    // NOT reach it — a dispatch here would mean decode silently dropped the
    // residual and changed the codec's numerics.
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 2, 8, 128, 24, true);
    // Asserted on the cache's own ring, not the process-global dispatch
    // counter: a concurrently-running test dispatching the kernel would make a
    // `delta == 0` assertion flake. The flash path cannot run without this
    // ring, so its absence is the race-free proof.
    assert!(
        !probe.gpu_ring_live,
        "rotor GPU ring was populated on a QJL store — decode would read a store \
         missing the QJL residual"
    );
}
