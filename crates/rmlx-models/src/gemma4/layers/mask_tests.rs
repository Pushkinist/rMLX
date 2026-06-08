//! Regression tests for the Gemma4 verify-step attention mask sizing.
//!
//! Issue #32: at a speculative verify-block step (`query_len > 1`, prompt long
//! enough to take the masked branch), the array-mode SWA / chunked-prefill mask
//! was sized from the model-wide `cache_base_offset` instead of the
//! cache-holding producer layer's own `offset()`. Across a partial-accept
//! rollback the two desync by one position, so the mask came out one key too
//! long (`(1,1,5,kv+1)`) vs the K the SDPA attended (`(1,8,5,kv)`), tripping the
//! opaque mlx-c `scaled_dot_product_attention` broadcast error.
//!
//! These tests pin the invariant the fix restores: for a verify-block step the
//! mask built from the *producer* offset has its key dim equal to
//! `producer_offset + seq` (== the post-update K seq dim), and the boundary
//! case where base_offset = producer_offset + 1 no longer inflates it.

use rmlx_mlx::Device;

use super::{build_attn_mask, LayerType};

/// FullAttention verify-block step at a long prompt (offset > 0, seq > 1) must
/// take the `"array"` masked branch and size the mask key dim to
/// `producer_offset + seq` — the exact post-update K length the cache-holding
/// layer's SDPA attends.
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test asserts on known-good shapes; unwrap/expect failures are the assertion"
)]
fn full_attn_verify_block_mask_matches_producer_k_len() {
    // Reproduce issue #32 magnitudes: prompt ~4121 already cached, verify block
    // of 5 tokens (b + 4 drafted). producer_offset is the producer cache's own
    // offset; K after update = producer_offset + seq.
    let producer_offset = 4121;
    let seq = 5;
    let window = 512usize; // irrelevant for FullAttention, still passed.

    let (mask, mode) = build_attn_mask(
        LayerType::FullAttention,
        seq,
        producer_offset, // effective_offset == producer_offset (non-rotating)
        producer_offset + seq,
        false, // attn_is_rotating
        window,
        Device::Cpu,
    )
    .unwrap();

    assert_eq!(
        mode, "array",
        "long-prompt verify block must use array mode"
    );
    let mask = mask.expect("array mode must carry a mask array");
    let shape = mask.shape();
    assert_eq!(shape.len(), 4, "mask is [1,1,seq,kv]");
    assert_eq!(shape[2], seq, "query dim");
    // The invariant: mask key dim == post-update K seq dim (producer_offset+seq).
    assert_eq!(
        shape[3],
        producer_offset + seq,
        "mask key dim must equal producer K seq dim (producer_offset + seq)"
    );
}

/// SlidingAttention verify-block step beyond the window (masked branch) must
/// size the mask key dim to `effective_offset + seq`, where effective_offset is
/// the producer's own offset capped at `window - 1` for a rotating cache. K
/// after a wrapped rotating update is `(window - 1) + seq`, so the mask must
/// match that, not `base_offset + seq`.
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test asserts on known-good shapes; unwrap/expect failures are the assertion"
)]
fn sliding_attn_verify_block_mask_matches_capped_k_len() {
    let window = 512usize;
    let seq = 5;
    // Rotating producer well past the window: K is window-capped, so
    // effective_offset = window - 1.
    let producer_offset = 4121;
    let effective_offset = producer_offset.min(window as i32 - 1);

    let (mask, mode) = build_attn_mask(
        LayerType::SlidingAttention,
        seq,
        effective_offset,
        effective_offset + seq,
        true, // attn_is_rotating
        window,
        Device::Cpu,
    )
    .unwrap();

    assert_eq!(mode, "array", "SWA prefill/verify block uses array mode");
    let mask = mask.expect("array mode must carry a mask array");
    let shape = mask.shape();
    assert_eq!(shape[2], seq, "query dim");
    assert_eq!(
        shape[3],
        effective_offset + seq,
        "SWA verify-block mask key dim must equal capped K seq dim"
    );
    // Capped K is far smaller than the model-wide base offset — the bug would
    // have sized this from base_offset and broadcast-failed against the ring K.
    assert!(
        shape[3] < producer_offset,
        "rotating K is window-capped, well below the absolute offset"
    );
}

/// The off-by-one the fix removes: had the mask been sized from a base_offset
/// that is producer_offset + 1 (the verify-block rollback desync), its key dim
/// would exceed the producer K seq dim by exactly one — the #32 crash shape.
/// This asserts the producer-sized mask does NOT exhibit that, and documents
/// the failing shape for clarity.
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test asserts on known-good shapes; unwrap/expect failures are the assertion"
)]
fn producer_sized_mask_avoids_base_offset_off_by_one() {
    let seq = 5;
    let producer_offset = 4121;
    let base_offset_desynced = producer_offset + 1; // the round-rollback drift.

    // Correct (producer-sized) mask.
    let (mask_ok, _) = build_attn_mask(
        LayerType::FullAttention,
        seq,
        producer_offset,
        producer_offset + seq,
        false,
        512,
        Device::Cpu,
    )
    .unwrap();
    let k_seq = producer_offset + seq; // post-update K length.
    assert_eq!(
        mask_ok.unwrap().shape()[3],
        k_seq,
        "producer-sized mask matches K"
    );

    // What the old base_offset-sized mask would have produced: kv = K + 1.
    let (mask_bad, _) = build_attn_mask(
        LayerType::FullAttention,
        seq,
        base_offset_desynced,
        base_offset_desynced + seq,
        false,
        512,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(
        mask_bad.unwrap().shape()[3],
        k_seq + 1,
        "base_offset-sized mask is one key too long — the #32 broadcast crash"
    );
}
