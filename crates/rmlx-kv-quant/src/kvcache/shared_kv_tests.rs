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

/// `materialise` must return K at the **stream dtype**, never a raw f32 dequant.
///
/// The pair goes to a separate downstream model's attention, so a K wider than
/// V promotes that model's whole stream — the KV-promotion class this codebase
/// has closed repeatedly. The rotor store dequantises through a `Vec<f32>`, so
/// K arrives F32 and the cast to V's dtype is load-bearing: drop it and a bf16
/// stream silently gets an f32 K.
///
/// Pinned against a **bf16** producer specifically, because that is the
/// production shape — the other tests here push F32 K/V, which would hide the
/// bug (F32 K matches an F32 V by accident).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test asserts on a known rank-4 shape"
)]
fn materialise_casts_k_to_the_stream_dtype() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let mut cache = seeded_cache();

    // bf16 K/V, as a real model pushes.
    let bf16 = |data: &[f32], shape: &[i32]| -> Array {
        f32_array(data, shape)
            .astype(Dtype::Bf16, device)
            .expect("cast to bf16")
    };
    let pf = (PREFILL * KV_H * HEAD_DIM) as usize;
    let k = bf16(&lcg_data(pf, 1), &[1, KV_H, PREFILL, HEAD_DIM]);
    let v = bf16(&lcg_data(pf, 2), &[1, KV_H, PREFILL, HEAD_DIM]);
    let q = bf16(
        &lcg_data((PREFILL * N_Q_HEADS * HEAD_DIM) as usize, 3),
        &[1, N_Q_HEADS, PREFILL, HEAD_DIM],
    );
    cache
        .update_and_sdpa_shared_source(&q, &k, &v, scale(), "causal", None, device)
        .expect("bf16 prefill");
    cache.exit_prefill(device).expect("exit_prefill");

    let one = (KV_H * HEAD_DIM) as usize;
    let k1 = bf16(&lcg_data(one, 11), &[1, KV_H, 1, HEAD_DIM]);
    let v1 = bf16(&lcg_data(one, 21), &[1, KV_H, 1, HEAD_DIM]);
    let q1 = bf16(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 31),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    let (out, share) = cache
        .update_and_sdpa_shared_source(&q1, &k1, &v1, scale(), "", None, device)
        .expect("bf16 decode");
    out.eval().expect("out eval");
    assert!(
        matches!(share, SharedKv::Store { .. }),
        "precondition: this must be a store-backed share, or the test proves nothing"
    );

    let (k_m, v_m) = share
        .materialise(Some(&cache), device)
        .expect("materialise a store-backed share");
    assert_eq!(
        v_m.dtype(),
        Dtype::Bf16,
        "precondition: the V mirror carries the bf16 stream dtype"
    );
    assert_eq!(
        k_m.dtype(),
        Dtype::Bf16,
        "materialised K must follow the stream dtype — a raw f32 dequant here \
         promotes the consumer model's attention stream to f32"
    );
    assert_eq!(k_m.dtype(), v_m.dtype(), "K and V must agree on dtype");
    assert_eq!(
        k_m.shape()[2],
        share.kv_len().expect("kv_len"),
        "materialised K must span the shared length"
    );
}

/// `check_shared_kv_len` is the guard between a producer/consumer length desync
/// and a silently-wrong attended prefix: a consumer that reads one key too many
/// or too few gets a plausible-looking answer, not an error. Pin both
/// directions against a live store — stubbing the guard to `Ok(())` must turn
/// this red.
#[test]
fn store_share_refuses_length_desync_against_live_store() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let mut cache = prefilled_producer();
    let share = producer_decode_step(&mut cache, 0);
    let live = share.kv_len().expect("kv_len");

    let q = f32_array(
        &lcg_data((N_Q_HEADS * HEAD_DIM) as usize, 77),
        &[1, N_Q_HEADS, 1, HEAD_DIM],
    );
    // Sanity: the honest length must succeed, or the negatives below prove
    // nothing (they would fail for an unrelated reason).
    cache
        .sdpa_shared(&q, scale(), None, live, device)
        .expect("the live store length must be accepted");

    for wrong in [live + 1, live - 1] {
        assert!(
            cache.sdpa_shared(&q, scale(), None, wrong, device).is_err(),
            "sdpa_shared accepted kv_len={wrong} against a store holding {live} — a \
             desync must error, not attend the wrong prefix"
        );
        assert!(
            cache.materialise_shared_kv(wrong, device).is_err(),
            "materialise_shared_kv accepted kv_len={wrong} against a store \
             holding {live} — a desync must error"
        );
    }
}

/// A store-backed share must never be silently answered from a cache that does
/// not hold it: no producer supplied, or a codec with no fused-over-store path.
#[test]
fn store_share_refuses_missing_producer_and_wrong_codec() {
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
        share.materialise(None, device).is_err(),
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
