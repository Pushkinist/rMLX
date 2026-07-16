//! Cross-layer KV sharing must not defeat the fused decode kernels.
//!
//! A producer layer in a shared-KV topology routes through
//! `update_and_sdpa_shared_source`. If that chain lacks the fused-over-store
//! arms `update_and_sdpa` has, the producer is pushed onto the legacy bf16 path
//! and **every** fused decode kernel is silently dead for **every** model with
//! KV sharing — regardless of arch or codec. These tests pin the two halves of
//! the contract:
//!
//! 1. a shared-KV producer reaches the same kernel a non-sharing model reaches;
//! 2. a consumer attends the producer's quant store through that same kernel,
//!    rather than forcing a bf16 materialisation.
//!
//! # No env dependence
//!
//! Like `rotor_flash_dispatch_tests`, these seed the rotor store with a
//! pre-built rotor table, which pins the codec's QJL decision for the store's
//! lifetime independently of the process-global toggle.

use super::{KvCache, SharedKv};
use crate::clifford::make_rotor_table;
use crate::quant::KvQuant;
use crate::rotor_flash_decode_msl::rotor_flash_decode_dispatch_count;
use crate::rotorquant::n_groups_for;
use crate::storage::{KvStorage, QuantRotorK3};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

const MAX_SEQ: i32 = 512;
const KV_H: i32 = 2;
const N_Q_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
const PREFILL: i32 = 24;

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Rotor3 K-only cache with QJL pinned off, so the flash kernel is eligible.
fn seeded_cache() -> KvCache {
    let rotors = make_rotor_table(0, 0, n_groups_for(HEAD_DIM as usize));
    let storage = KvStorage::RotorKOnly3 {
        k: Some(QuantRotorK3::from_cpu_blocks(
            rotors,
            None,
            Vec::new(),
            vec![1, KV_H, 0, HEAD_DIM],
            MAX_SEQ,
            0,
        )),
        max_seq: MAX_SEQ,
    };
    KvCache::from_storage(storage, KvQuant::RotorKOnly3, 0, 0)
}

fn scale() -> f32 {
    1.0 / (HEAD_DIM as f32).sqrt()
}

/// Prefill a producer cache through the shared-source entry point and leave it
/// decode-ready, exactly as a cross-layer-KV arch does.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn prefilled_producer() -> KvCache {
    let device = Device::Gpu;
    let mut cache = seeded_cache();
    let pf = (PREFILL * KV_H * HEAD_DIM) as usize;
    let k = f32_array(&lcg_data(pf, 1), &[1, KV_H, PREFILL, HEAD_DIM]);
    let v = f32_array(&lcg_data(pf, 2), &[1, KV_H, PREFILL, HEAD_DIM]);
    let q = f32_array(
        &lcg_data((PREFILL * N_Q_HEADS * HEAD_DIM) as usize, 3),
        &[1, N_Q_HEADS, PREFILL, HEAD_DIM],
    );
    let (_out, share) = cache
        .update_and_sdpa_shared_source(&q, &k, &v, scale(), "causal", None, device)
        .expect("prefill update_and_sdpa_shared_source");
    // Prefill is not decode-only, so no fused arm may fire: the share is bf16.
    assert!(
        matches!(share, SharedKv::Bf16(_, _)),
        "a prefill chunk must yield a bf16 share — the fused arms are decode-only"
    );
    cache.exit_prefill(device).expect("exit_prefill");
    cache
}

/// One decode step on the producer. Returns the share it offers consumers.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn producer_decode_step(cache: &mut KvCache, step: u64) -> SharedKv {
    let device = Device::Gpu;
    let one = (KV_H * HEAD_DIM) as usize;
    let k1 = f32_array(&lcg_data(one, 10 + step), &[1, KV_H, 1, HEAD_DIM]);
    let v1 = f32_array(&lcg_data(one, 20 + step), &[1, KV_H, 1, HEAD_DIM]);
    let q1 = f32_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 30 + step),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    let (out, share) = cache
        .update_and_sdpa_shared_source(&q1, &k1, &v1, scale(), "", None, device)
        .expect("decode update_and_sdpa_shared_source");
    out.eval().expect("producer out eval");
    share
}

/// The defect: a shared-KV producer must reach the same fused kernel a
/// non-sharing model reaches. Before the fix this chain had no rotor arm at all
/// and every step fell through to `update()`'s O(seq) CPU dequant, so the
/// dispatch delta was 0 no matter what the codec supported.
#[test]
fn shared_source_producer_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let mut cache = prefilled_producer();
    let before = rotor_flash_decode_dispatch_count();
    for step in 0..4_u64 {
        let share = producer_decode_step(&mut cache, step);
        assert!(
            matches!(share, SharedKv::Store { .. }),
            "a fused decode step must share the quant store, not a bf16 dequant"
        );
    }
    let delta = rotor_flash_decode_dispatch_count() - before;
    assert!(
        delta >= 4,
        "update_and_sdpa_shared_source did not reach the rotor flash kernel on all 4 \
         decode steps (delta={delta}) — a shared-KV topology is still forcing the \
         codec onto the legacy bf16 path"
    );
}

/// A consumer attends the producer's store through the same kernel — the
/// producer never materialises bf16 K/V on its behalf.
#[test]
fn shared_consumer_attends_store_through_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let mut cache = prefilled_producer();
    let share = producer_decode_step(&mut cache, 0);

    // Consumer Q: its own projection, same shape contract as the producer's.
    let q_c = f32_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 99),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    let before = rotor_flash_decode_dispatch_count();
    let out = share
        .sdpa(Some(&cache), &q_c, scale(), "", None, device)
        .expect("consumer sdpa over the producer's store");
    out.eval().expect("consumer out eval");
    let delta = rotor_flash_decode_dispatch_count() - before;

    assert!(
        delta >= 1,
        "consumer SDPA did not reach the flash kernel (delta={delta}) — it is \
         reading a dequantized bf16 tensor instead of the producer's store"
    );
    assert_eq!(
        out.shape(),
        vec![1, N_Q_HEADS, 1, HEAD_DIM],
        "consumer output shape"
    );
}

/// The consumer's mask is sized from the **producer's** K length. A store-backed
/// share must report exactly what a bf16 share's `k.shape()[2]` would have.
#[test]
fn store_share_kv_len_tracks_producer_offset() {
    if skip_if_no_gpu_env() {
        return;
    }
    let mut cache = prefilled_producer();
    for step in 0..3_u64 {
        let share = producer_decode_step(&mut cache, step);
        assert_eq!(
            share.kv_len().expect("kv_len"),
            cache.offset(),
            "share kv_len must equal the producer's post-update offset — a \
             consumer sizes its mask from it"
        );
    }
}

/// A store-backed share must never be silently answered from a cache that does
/// not hold it. Both the missing-producer and the length-desync cases error.
#[test]
fn store_share_refuses_missing_producer_and_length_desync() {
    let device = Device::Gpu;
    let share = SharedKv::Store { kv_len: 8 };
    let q = f32_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 7),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    assert!(
        share.sdpa(None, &q, scale(), "", None, device).is_err(),
        "a store-backed share with no producer cache must error, not guess"
    );
    assert!(
        share.materialise_bf16(None, device).is_err(),
        "materialising a store-backed share with no producer cache must error"
    );

    // A non-store codec must refuse a store-backed consumer read outright
    // rather than answer with some other codec's numbers.
    let cache = KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ);
    assert!(
        cache.sdpa_shared(&q, scale(), None, 8, device).is_err(),
        "sdpa_shared on a storage variant with no fused-over-store path must error"
    );
}

/// `KvQuant::None` keeps the bf16 contract: the share is the stored tensors and
/// consumers read them exactly as before.
#[test]
fn none_quant_shares_bf16_unchanged() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ);
    let one = (KV_H * HEAD_DIM) as usize;
    let k1 = f32_array(&lcg_data(one, 1), &[1, KV_H, 1, HEAD_DIM]);
    let v1 = f32_array(&lcg_data(one, 2), &[1, KV_H, 1, HEAD_DIM]);
    let q1 = f32_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 3),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    let (_out, share) = cache
        .update_and_sdpa_shared_source(&q1, &k1, &v1, scale(), "", None, device)
        .expect("None-quant shared source");
    match &share {
        SharedKv::Bf16(k, v) => {
            assert_eq!(k.shape(), vec![1, KV_H, 1, HEAD_DIM], "shared K shape");
            assert_eq!(v.shape(), vec![1, KV_H, 1, HEAD_DIM], "shared V shape");
        }
        SharedKv::Store { .. } => {
            panic!("KvQuant::None has no quant store to share — must stay bf16")
        }
    }
    assert_eq!(share.kv_len().expect("kv_len"), 1, "bf16 share kv_len");
    let out = share
        .sdpa(Some(&cache), &q1, scale(), "", None, device)
        .expect("bf16 consumer sdpa");
    out.eval().expect("bf16 consumer eval");
}
