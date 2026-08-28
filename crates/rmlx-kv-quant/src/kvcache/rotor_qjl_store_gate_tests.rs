//! The rotor K-only decode update gates its GPU encode on the **store's**
//! sticky QJL flag, not the process-global `RMLX_ROTOR_QJL` env.
//!
//! A rotor codec fixes its QJL decision at first append; the bytes it holds are
//! written under that decision and must be read back the same way. A later env
//! toggle governs only *newly created* stores — it must never reinterpret a
//! store already on disk. The sdpa fused fast path already reads the store flag
//! (`rotor_k_store_uses_qjl`); this proves the `update()` decode path agrees.
//!
//! # Why env-manipulating (and why its own file)
//!
//! The sibling `rotor_flash_dispatch_tests.rs` is deliberately env-free. This
//! test is the opposite: it *seeds a QJL-off store, flips the env ON, and asks
//! the update path which way it goes*. The store must win. Env writes are
//! serialized on the shared test env lock.
//!
//! # Mutation contract
//!
//! Revert the fix (gate on `rotor_qjl_enabled()` instead of the store) and this
//! test MUST fail: with the env ON the reverted gate takes the CPU append and
//! never allocates the GPU ring, so the `gpu.is_allocated()` witness flips to
//! `false`. A test that survives the mutation proves nothing.
#![allow(unsafe_code)]

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::quant::KvQuant;
use crate::rotor_flash_decode_msl::rotor_flash_decode_dispatch_count;
use crate::rotorquant::n_groups_for;
use crate::storage::{KvStorage, QuantRotorK3, QuantRotorK4};
use crate::test_utils::{env_lock, lcg_data, skip_if_no_gpu_env};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

const MAX_SEQ: i32 = 512;

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Build a rotor K-only cache whose store carries **no** QJL sideband,
/// independent of the process env. Pre-seeding a non-empty rotor table pins the
/// decision: the append paths only touch `qjl_s_matrix` inside their
/// `rotors.is_empty()` lazy-init branch, so a store handed a rotor table keeps
/// the `None` QJL matrix it was built with.
fn seeded_qjl_off_cache(quant: KvQuant, kv_h: i32, head_dim: i32) -> KvCache {
    let n_groups = n_groups_for(head_dim as usize);
    let rotors = make_rotor_table(0, 0, n_groups);
    let shape = vec![1_i32, kv_h, 0, head_dim];
    let storage = if quant == KvQuant::RotorKOnly4 {
        KvStorage::RotorKOnly4 {
            k: Some(QuantRotorK4::from_cpu_blocks(
                rotors,
                None,
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
                None,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        }
    };
    KvCache::from_storage(storage, quant, 0, 0, DispatchPolicy::default(), false)
}

/// True once the cache's rotor K store has a live GPU ring — the witness that
/// the update path took the QJL-off (GPU encode) branch. The CPU append never
/// allocates the ring, so its absence is the QJL-on (env-gated) branch.
fn store_ring_live(cache: &KvCache) -> bool {
    if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else {
        false
    }
}

/// Seed a QJL-off store, flip the env ON, drive one decode step through the
/// public `update()` path, and assert the store's decision won (GPU ring live).
#[allow(
    clippy::expect_used,
    reason = "test: a failed append/lock is a test failure with a clear message"
)]
fn store_flag_beats_env(quant: KvQuant) {
    let kv_h = 2_i32;
    let head_dim = 128_i32;

    let _guard = env_lock();
    // The CLI override shadows the env; if some other test installed it, the env
    // flip below is a no-op and the test cannot prove the store beats the env.
    if crate::rotor_qjl::rotor_qjl_cli_is_set() {
        return;
    }
    // SAFETY: env lock held — no concurrent env reader/writer.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };
    // Precondition: the env must actually read ON, otherwise the mutant (which
    // gates on the env) would also take the GPU branch and survive.
    assert!(
        crate::rotor_qjl::rotor_qjl_enabled(),
        "env must read ON for this test to distinguish store from env"
    );

    let mut cache = seeded_qjl_off_cache(quant, kv_h, head_dim);
    let one = (kv_h * head_dim) as usize;
    let k1 = f32_array(&lcg_data(one, 11), &[1, kv_h, 1, head_dim]);
    let v1 = f32_array(&lcg_data(one, 22), &[1, kv_h, 1, head_dim]);
    let (k_full, v_full) = cache
        .update(&k1, &v1, Device::Gpu)
        .expect("decode update on QJL-off store");
    k_full.eval().expect("k eval");
    v_full.eval().expect("v eval");

    // SAFETY: env lock still held.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };

    assert!(
        store_ring_live(&cache),
        "update path gated on the env instead of the store: a QJL-off store took \
         the CPU append while the env said ON, so the GPU ring was never \
         allocated"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_qjl_store_gate -- --ignored --test-threads=1"]
fn rotor3_update_gates_on_store_flag_not_env() {
    if skip_if_no_gpu_env() {
        return;
    }
    store_flag_beats_env(KvQuant::RotorKOnly3);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_qjl_store_gate -- --ignored --test-threads=1"]
fn rotor4_update_gates_on_store_flag_not_env() {
    if skip_if_no_gpu_env() {
        return;
    }
    store_flag_beats_env(KvQuant::RotorKOnly4);
}

/// Dispatch witness for the stock default: a rotor K-only cache built with no
/// env override and no CLI install (i.e. `rotor_qjl_enabled()` returns its
/// default) must reach the fused Metal flash-decode kernel on decode. If the
/// default were on, the codec would fall onto the CPU path and never dispatch.
///
/// Model-agnostic: the kernel keys off codec + shape, not an arch. This proves
/// the default choice at the codec layer; the two-arch real-model measurement
/// (Bonsai + gemma) lives in the PR description.
#[allow(
    clippy::expect_used,
    reason = "test: a failed append/lock/eval is a test failure with a clear message"
)]
fn default_reaches_fused_kernel(quant: KvQuant) {
    let kv_h = 2_i32;
    let n_q_heads = 8_i32;
    let head_dim = 128_i32;
    let prefill = 24_i32;
    let device = Device::Gpu;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let _guard = env_lock();
    if crate::rotor_qjl::rotor_qjl_cli_is_set() {
        return;
    }
    // Stock defaults: no env override.
    // SAFETY: env lock held — no concurrent env reader/writer.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
    // The whole point of this test is that the *default* keeps QJL off so the
    // fused path is reachable. Assert the default here so a future re-flip to
    // on-by-default fails loudly rather than silently disabling the kernel.
    assert!(
        !crate::rotor_qjl::rotor_qjl_enabled(),
        "stock default must leave QJL off so the rotor fused Metal path is reachable"
    );

    // Build the cache the production way — let the default decide QJL, do not
    // pre-seed the store's flag.
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ);
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
    }
    let delta = rotor_flash_decode_dispatch_count() - before;

    assert!(
        store_ring_live(&cache),
        "default-built rotor cache has no live GPU ring — the fused path was not reached"
    );
    assert!(
        delta >= 4,
        "default-built rotor cache did not dispatch the fused Metal flash-decode \
         kernel on all 4 decode steps (delta={delta}); at stock defaults the \
         rotor K-only decode must be on the Metal path, not CPU"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_qjl_store_gate -- --ignored --test-threads=1"]
fn rotor3_default_reaches_fused_metal_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    default_reaches_fused_kernel(KvQuant::RotorKOnly3);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_qjl_store_gate -- --ignored --test-threads=1"]
fn rotor4_default_reaches_fused_metal_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    default_reaches_fused_kernel(KvQuant::RotorKOnly4);
}
