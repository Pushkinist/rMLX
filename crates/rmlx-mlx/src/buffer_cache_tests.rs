//! Tests for [`trim_buffer_cache`]: which cached buffers it frees.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::*;

/// Elements of the older buffer: 4 MiB of f32, whole 16 KiB pages.
const OLD_ELEMENTS: usize = 1 << 20;

/// Elements of the newer buffer: 16 MiB of f32. The size is far enough from
/// the older one that MLX never serves a request for one from a buffer of the
/// other.
const RECENT_ELEMENTS: usize = 4 << 20;

fn bytes_of(elements: usize) -> u64 {
    (elements * 4) as u64
}

fn zeros(elements: usize) -> Array {
    Array::from_f32_slice(&vec![0.0; elements], &[elements as i32]).unwrap()
}

/// A trim to the size of the newest cached buffer frees the older buffer and
/// keeps the newest one, which the next request of its size reuses.
///
/// A full clear frees both. A trim that frees from the newest end frees the
/// 16 MiB buffer and keeps the 4 MiB one.
#[test]
#[ignore = "GPU Metal context — run with `make gpu-test`"]
fn trim_frees_the_oldest_buffer_and_keeps_the_newest() {
    // The cache under test is the allocator of the Metal device, so run one
    // GPU op to have the device up. Its buffers are gone before the cache is
    // cleared.
    let one = zeros(1);
    add(&one, &one, Device::Gpu).unwrap().eval().unwrap();
    drop(one);
    // A completion handler frees a temporary of the evaluation; wait for it so
    // that it cannot put a buffer in the cache after the clear.
    synchronize_gpu().unwrap();
    assert!(mlx_clear_cache());

    let old = zeros(OLD_ELEMENTS);
    let recent = zeros(RECENT_ELEMENTS);
    drop(old);
    drop(recent);
    let recent_bytes = bytes_of(RECENT_ELEMENTS);
    assert_eq!(
        mlx_cache_memory_bytes().unwrap(),
        bytes_of(OLD_ELEMENTS) + recent_bytes,
        "no power: both buffers must be in the cache before the trim"
    );

    assert!(trim_buffer_cache(recent_bytes));
    assert_eq!(
        mlx_cache_memory_bytes().unwrap(),
        recent_bytes,
        "the trim must free the older buffer and keep the newer one"
    );

    let reused = zeros(RECENT_ELEMENTS);
    assert_eq!(
        mlx_cache_memory_bytes().unwrap(),
        0,
        "a request of the size of the kept buffer must be served from the cache"
    );
    drop(reused);
}
