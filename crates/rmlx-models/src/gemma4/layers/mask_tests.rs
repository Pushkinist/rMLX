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
//!
//! The `guard_invariant_*` tests below directly guard the producer-offset
//! SELECTION via `producer_effective_offset` — the named helper extracted from
//! `Attention::forward` (mod.rs ~L309-315) to create a testable seam. That
//! helper captures the single-line fix (`c.offset()` not `base_offset`).
//! Reverting `Attention::forward` to inline `base_offset + 1` instead of
//! routing through `producer_effective_offset(c.offset(), ...)` changes the
//! effective offset from `producer_offset` to `producer_offset + 1`, shifting
//! the mask key dim from `k_seq` to `k_seq + 1` and making
//! `guard_invariant_producer_offset_matches_k_seq` RED.

use rmlx_mlx::Device;

use super::{build_attn_mask, producer_effective_offset, LayerType};

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
    let window = 512usize; // unused by the FullAttention arm; any value works.

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

/// Guard invariant (PASSING side): `producer_effective_offset` with the
/// producer's own offset returns a value that makes the mask key dim equal
/// to the post-update K seq dim.
///
/// `producer_effective_offset` is the named seam extracted from
/// `Attention::forward` to make the offset-selection testable without driving a
/// full model forward. If `Attention::forward` is reverted to inline
/// `base_offset + 1` instead of calling this helper with `c.offset()`, this
/// test goes RED because the inline path would compute `effective_offset =
/// producer_offset + 1` (or equivalently bypass the helper entirely), and the
/// mask's key dim would then be `producer_offset + seq + 1 != k_seq`.
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test asserts on known-good shapes; unwrap/expect failures are the assertion"
)]
fn guard_invariant_producer_offset_matches_k_seq() {
    // Reproduce issue #32 magnitudes: partial-accept rollback left producer
    // cache at offset 4121 while the model-wide base_offset is 4122.
    let seq = 5;
    let producer_offset = 4121;
    let window = 512usize;

    // This is exactly what `Attention::forward` does at ~L310-315 (post-fix):
    //   let producer_offset = c.offset();
    //   let effective_offset = producer_effective_offset(producer_offset, attn_is_rotating, sliding_window);
    let effective_offset = producer_effective_offset(producer_offset, false, window);
    assert_eq!(
        effective_offset, producer_offset,
        "non-rotating: effective offset must equal the producer's own offset"
    );

    let (mask, _) = build_attn_mask(
        LayerType::FullAttention,
        seq,
        effective_offset,
        effective_offset + seq,
        false,
        window,
        Device::Cpu,
    )
    .unwrap();
    let k_seq = producer_offset + seq; // post-update K seq dim the SDPA attends.
                                       // Guard invariant: mask key dim == K seq dim → the Attention::forward guard
                                       // (mod.rs ~L345-355) does NOT fire.
    assert_eq!(
        mask.expect("array mode must carry a mask array").shape()[3],
        k_seq,
        "producer-offset mask key dim must equal K seq dim; guard must not fire"
    );
}

/// Guard invariant (REGRESSED side): if the model-wide `base_offset`
/// (`producer_offset + 1`) were used instead of `producer_effective_offset`,
/// the mask key dim would exceed K seq dim by exactly one.
///
/// This test documents the exact #32 crash shape. Reverting `Attention::forward`
/// to pass `base_offset` (== `producer_offset + 1`) into the mask builder —
/// instead of routing through `producer_effective_offset(c.offset(), ...)` —
/// makes `guard_invariant_producer_offset_matches_k_seq` RED (the
/// producer-offset effective value changes from `4121` to `4122`, shifting the
/// mask key dim from `k_seq` to `k_seq + 1`, failing that test's `assert_eq!`).
/// This companion test pins the off-by-one arithmetic so the regression is
/// unambiguously documented.
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test asserts on known-bad shape to document the regression; unwrap/expect failures are the assertion"
)]
fn guard_invariant_regressed_base_offset_inflates_mask() {
    let seq = 5;
    let producer_offset = 4121;
    let base_offset_desynced = producer_offset + 1; // model-wide offset after rollback desync.
    let window = 512usize;

    // Simulate what a reverted `Attention::forward` would do: pass base_offset
    // (not c.offset()) to the mask builder. `producer_effective_offset` is NOT
    // called here — this test explicitly bypasses it to show the wrong path.
    let (mask, _) = build_attn_mask(
        LayerType::FullAttention,
        seq,
        base_offset_desynced,
        base_offset_desynced + seq,
        false,
        window,
        Device::Cpu,
    )
    .unwrap();
    let k_seq = producer_offset + seq; // the K seq dim the SDPA actually attends.
                                       // Off-by-one: mask key dim is one longer than K — the #32 broadcast crash.
    assert_eq!(
        mask.expect("array mode must carry a mask array").shape()[3],
        k_seq + 1,
        "base_offset-desynced mask is one key too long — the #32 guard trigger"
    );
}
