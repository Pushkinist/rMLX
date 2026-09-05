//! `KvCache::truncate_to` fails exactly where `can_truncate_to` refuses.
//!
//! A rollback that cannot reach its target is the shape of defect this pair
//! exists to stop: the SWA ring used to keep its offset and its rejected keys
//! while every full-attention layer dropped theirs, and nothing said so.

use crate::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

/// Deterministic `[1, 1, s, 2]` f32 K/V pair.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn kv(s: i32, base: f32) -> Array {
    let mut data: Vec<f32> = Vec::with_capacity((s * 2) as usize);
    for p in 0..s {
        data.push(base + p as f32);
        data.push(base + p as f32 + 0.5);
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, &[1, 1, s, 2], Dtype::F32).unwrap()
}

/// A windowed layer that a block write left in temporal order rolls its
/// rejected tail back, and reports the rolled-back offset.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn a_windowed_layer_rolls_back_a_verify_block_tail() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::None, 512, Some(4));
    cache.update(&kv(6, 0.0), &kv(6, 100.0), device).unwrap();
    cache.update(&kv(4, 6.0), &kv(4, 106.0), device).unwrap();
    assert_eq!(cache.offset(), 10);

    let target = 7;
    assert!(
        cache.can_truncate_to(target),
        "a wrapped ring in temporal order must be able to give a block tail back"
    );
    cache
        .truncate_to(target)
        .expect("and truncate_to must do what can_truncate_to promised");
    assert_eq!(cache.offset(), target);
}

/// The same layer, left in rotated order by single-token decode writes, refuses
/// — and says which layer and how far it was asked to go back, because the
/// message is what a caller with no second route has to act on.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn a_rotated_windowed_layer_refuses_rather_than_keeping_the_rejected_keys() {
    let device = Device::Cpu;
    let mut cache =
        KvCache::with_quant_max_seq_window(KvQuant::None, 512, Some(4)).with_layer_idx(7);
    cache.update(&kv(6, 0.0), &kv(6, 100.0), device).unwrap();
    for step in 0..3 {
        let p = (6 + step) as f32;
        cache.update(&kv(1, p), &kv(1, 100.0 + p), device).unwrap();
    }
    let before = cache.offset();
    assert_eq!(before, 9);

    assert!(
        !cache.can_truncate_to(8),
        "a rotated ring cannot give a key back"
    );
    let err = cache
        .truncate_to(8)
        .expect_err("and truncate_to must refuse rather than no-op at the old offset");
    let text = err.to_string();
    assert!(text.contains("layer 7"), "{text}");
    assert!(text.contains("sliding-window ring"), "{text}");
    assert_eq!(
        cache.offset(),
        before,
        "a refused rollback leaves the cache where it was"
    );
}

/// A full-attention layer carries the whole sequence, so every prefix is
/// reachable — the predicate must not refuse one and strand the round.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn a_full_attention_layer_reaches_every_prefix() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::None, 512, None);
    cache.update(&kv(10, 0.0), &kv(10, 100.0), device).unwrap();
    for target in 0..=10 {
        let mut c = KvCache::with_quant_max_seq_window(KvQuant::None, 512, None);
        c.update(&kv(10, 0.0), &kv(10, 100.0), device).unwrap();
        assert!(c.can_truncate_to(target), "target={target}");
        c.truncate_to(target).expect("full attention rolls back");
        assert_eq!(c.offset(), target);
    }
    assert!(
        !cache.can_truncate_to(11),
        "a target past the offset is not a prefix of anything"
    );
}
