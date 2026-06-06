//! Tests for [`MultimodalCache`] + key recipes.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::clone_on_ref_ptr,
    reason = "unit-test scaffolding: panics surface assertion failures; Arc clones are intentional for the concurrency test"
)]

use super::*;
use rmlx_mlx::{Array, Dtype};
use std::sync::Arc;

fn make_array(elems: i32) -> Array {
    let data = vec![0u8; (elems as usize) * 4];
    Array::from_bytes(&data, &[elems], Dtype::F32).expect("Array::from_bytes")
}

#[test]
fn image_key_same_pixels_same_hash() {
    let pixels = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let k1 = MmCacheKey::image_key(&pixels, 64, 64, 3, MmDtype::F32);
    let k2 = MmCacheKey::image_key(&pixels, 64, 64, 3, MmDtype::F32);
    assert_eq!(k1, k2);
}

#[test]
fn image_key_different_dims_different_hash() {
    // Same byte stream, different declared (H, W). Must NOT collide.
    let pixels = vec![0u8; 64];
    let k1 = MmCacheKey::image_key(&pixels, 8, 8, 1, MmDtype::F32);
    let k2 = MmCacheKey::image_key(&pixels, 4, 16, 1, MmDtype::F32);
    assert_ne!(
        k1, k2,
        "(H,W) reshape collision: header is supposed to disambiguate"
    );
}

#[test]
fn image_key_dtype_disambiguates() {
    let pixels = vec![7u8; 16];
    let kf = MmCacheKey::image_key(&pixels, 4, 4, 1, MmDtype::F32);
    let kb = MmCacheKey::image_key(&pixels, 4, 4, 1, MmDtype::Bf16);
    assert_ne!(kf, kb);
}

#[test]
fn audio_key_sr_disambiguates() {
    let pcm = vec![0u8; 32];
    let k16 = MmCacheKey::audio_key(&pcm, 16_000, MmDtype::F32, 1);
    let k48 = MmCacheKey::audio_key(&pcm, 48_000, MmDtype::F32, 1);
    assert_ne!(k16, k48, "sample-rate must change the digest");
}

#[test]
fn audio_key_same_inputs_same_hash() {
    let pcm = vec![3u8; 40];
    let a = MmCacheKey::audio_key(&pcm, 16_000, MmDtype::F32, 1);
    let b = MmCacheKey::audio_key(&pcm, 16_000, MmDtype::F32, 1);
    assert_eq!(a, b);
}

#[test]
fn disabled_cache_is_noop() {
    let c = MultimodalCache::new(0);
    assert!(c.is_disabled());
    let key = MmCacheKey::image_key(b"abc", 1, 1, 1, MmDtype::F32);
    let arr = make_array(4);
    let sz = array_byte_size(&arr).expect("array_byte_size");
    c.put(key, arr, sz);
    assert!(c.get(&key).is_none());
    let s = c.stats();
    assert_eq!(s.used_bytes, 0);
    assert_eq!(s.entries, 0);
    assert_eq!(s.capacity_bytes, 0);
}

#[test]
fn get_miss_then_hit_after_put() {
    let c = MultimodalCache::new(1 << 20);
    let key = MmCacheKey::image_key(b"px", 1, 1, 1, MmDtype::F32);
    assert!(c.get(&key).is_none());
    let arr = make_array(8);
    let sz = array_byte_size(&arr).expect("array_byte_size");
    c.put(key, arr, sz);
    let got = c.get(&key).expect("hit after put");
    assert_eq!(got.shape(), vec![8]);
    let s = c.stats();
    assert_eq!(s.hits, 1);
    assert_eq!(s.misses, 1);
    assert_eq!(s.entries, 1);
    assert!(s.used_bytes >= sz);
}

#[test]
fn lru_evicts_to_budget() {
    // Budget for exactly two 32-byte entries.
    let elems_per_entry = 8_i32; // 32 bytes f32
    let arr_bytes = (elems_per_entry as usize) * 4;
    let budget = arr_bytes * 2;
    let c = MultimodalCache::new(budget);

    let k1 = MmCacheKey::image_key(b"k1", 1, 1, 1, MmDtype::F32);
    let k2 = MmCacheKey::image_key(b"k2", 1, 1, 1, MmDtype::F32);
    let k3 = MmCacheKey::image_key(b"k3", 1, 1, 1, MmDtype::F32);
    c.put(k1, make_array(elems_per_entry), arr_bytes);
    c.put(k2, make_array(elems_per_entry), arr_bytes);
    // Touch k1 so k2 becomes LRU.
    let _ = c.get(&k1);
    c.put(k3, make_array(elems_per_entry), arr_bytes);

    assert!(c.get(&k1).is_some(), "k1 was just touched, should survive");
    assert!(c.get(&k3).is_some(), "k3 is the newest entry");
    assert!(c.get(&k2).is_none(), "k2 was LRU and should be evicted");

    let s = c.stats();
    assert_eq!(s.entries, 2);
    assert!(s.used_bytes <= s.capacity_bytes);
}

#[test]
fn put_oversize_is_noop() {
    let c = MultimodalCache::new(16);
    let arr = make_array(64); // 256 bytes
    let sz = array_byte_size(&arr).expect("array_byte_size");
    let key = MmCacheKey::image_key(b"big", 1, 1, 1, MmDtype::F32);
    c.put(key, arr, sz);
    assert!(c.get(&key).is_none(), "oversize entry must not be cached");
    let s = c.stats();
    assert_eq!(s.entries, 0);
    assert_eq!(s.used_bytes, 0);
}

#[test]
fn stats_track_hits_misses() {
    let c = MultimodalCache::new(1 << 20);
    let key = MmCacheKey::image_key(b"x", 2, 2, 1, MmDtype::F32);
    // miss x3
    for _ in 0..3 {
        let _ = c.get(&key);
    }
    let arr = make_array(4);
    let sz = array_byte_size(&arr).expect("array_byte_size");
    c.put(key, arr, sz);
    // hit x2
    for _ in 0..2 {
        let _ = c.get(&key);
    }
    let s = c.stats();
    assert_eq!(s.hits, 2);
    assert_eq!(s.misses, 3);
}

#[test]
fn clear_drops_entries_keeps_counters() {
    let c = MultimodalCache::new(1 << 20);
    let key = MmCacheKey::image_key(b"y", 2, 2, 1, MmDtype::F32);
    let _ = c.get(&key);
    let arr = make_array(4);
    let sz = array_byte_size(&arr).expect("array_byte_size");
    c.put(key, arr, sz);
    c.clear();
    let s = c.stats();
    assert_eq!(s.entries, 0);
    assert_eq!(s.used_bytes, 0);
    assert_eq!(s.misses, 1);
}

#[test]
fn concurrent_get_put_safe() {
    let c = Arc::new(MultimodalCache::new(1 << 20));
    let mut handles = Vec::new();
    for t in 0..8 {
        let c = c.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0..32 {
                let buf = [t as u8, i as u8, 0, 0];
                let key = MmCacheKey::image_key(&buf, 1, 1, 1, MmDtype::F32);
                if c.get(&key).is_none() {
                    let arr = make_array(2);
                    let sz = array_byte_size(&arr).expect("array_byte_size");
                    c.put(key, arr, sz);
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread join");
    }
    let s = c.stats();
    // Best-effort sanity: at least some hits, used_bytes within budget.
    assert!(s.entries > 0);
    assert!(s.used_bytes <= s.capacity_bytes);
}

#[test]
fn array_byte_size_matches_dtype_and_shape() {
    let a = make_array(10);
    assert_eq!(array_byte_size(&a).expect("array_byte_size"), 10 * 4);
}

#[test]
fn key_short_hex_is_8_chars() {
    let k = MmCacheKey::image_key(b"abc", 1, 1, 1, MmDtype::F32);
    assert_eq!(k.short_hex().len(), 8);
}
