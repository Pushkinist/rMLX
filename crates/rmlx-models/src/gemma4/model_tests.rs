//! Model-free guards for the Gemma4 vision bidirectional-attention overlay.
//!
//! Gemma 4 conditions each image's soft tokens with bidirectional attention
//! (every soft token of an image attends to every other soft token of that
//! image). The encoder-free unified embedder produces raw projected patches
//! with no pre-integrated context, so reading them causally mis-conditions the
//! decoder (chromatic colours misnamed, spatial layout hallucinated). These
//! tests pin the overlay shape and the exact allow/block pattern so that
//! regression to a causal-only image block is caught without a model.

use rmlx_mlx::{Array, Device, Dtype};

use super::build_vision_bidi_overlay_for_test as build_vision_bidi_overlay;

const BOI: i32 = 255_999;
const EOI: i32 = 258_882;
const IMG: i32 = 258_880;

/// Read the `[1,1,seq,seq]` overlay to a host `f32` grid for assertions.
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test asserts the overlay materialises; chunks_exact(4) yields 4-byte slabs"
)]
fn overlay_grid(ids: &[i32]) -> Vec<f32> {
    let seq = ids.len() as i32;
    let arr = Array::from_i32_slice(ids, &[seq]).unwrap();
    let m = build_vision_bidi_overlay(&arr, seq, Device::Cpu).expect("overlay present");
    let raw = m
        .astype(Dtype::F32, Device::Cpu)
        .unwrap()
        .to_bytes()
        .unwrap();
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Image soft tokens inside one `<boi> .. <eoi>` block attend bidirectionally to
/// each other; everything else (text↔text, text↔image, the markers themselves)
/// is blocked by the overlay (the causal mask supplies those allowances).
#[test]
#[allow(
    clippy::indexing_slicing,
    clippy::float_cmp,
    reason = "fixed-size test grid indexed by construction; overlay cells are exact 0.0 / -1e30"
)]
fn overlay_opens_image_block_bidirectionally() {
    // [BOS, BOI, IMG, IMG, IMG, EOI, text]
    let ids = [2, BOI, IMG, IMG, IMG, EOI, 100];
    let n = ids.len();
    let grid = overlay_grid(&ids);

    // Soft-token positions are indices 2,3,4 (strictly between BOI@1 and EOI@5).
    let soft = [2usize, 3, 4];
    for &i in &soft {
        for &j in &soft {
            // Bidirectional: allowed (0.0) for every ordered pair, including j>i.
            assert_eq!(grid[i * n + j], 0.0, "soft ({i},{j}) must be open");
        }
    }
    // A future-looking soft pair (j>i) is the discriminating case: under a
    // causal-only mask this would be blocked.
    assert_eq!(
        grid[2 * n + 4],
        0.0,
        "soft token must see a later soft token"
    );

    // Text/markers stay closed in the overlay (causal mask handles them).
    for i in 0..n {
        for j in 0..n {
            let both_soft = soft.contains(&i) && soft.contains(&j);
            if !both_soft {
                assert!(
                    grid[i * n + j] < -1.0e9,
                    "non-soft pair ({i},{j}) must be blocked by the overlay"
                );
            }
        }
    }
}

/// Two image blocks each open only within themselves — soft tokens of image A
/// never attend to soft tokens of image B.
#[test]
#[allow(
    clippy::indexing_slicing,
    clippy::float_cmp,
    reason = "fixed-size test grid indexed by construction; overlay cells are exact 0.0 / -1e30"
)]
fn overlay_does_not_cross_image_blocks() {
    // [BOS, BOI, IMG, IMG, EOI, BOI, IMG, IMG, EOI]
    let ids = [2, BOI, IMG, IMG, EOI, BOI, IMG, IMG, EOI];
    let n = ids.len();
    let grid = overlay_grid(&ids);

    let block_a = [2usize, 3];
    let block_b = [6usize, 7];
    for &i in &block_a {
        for &j in &block_a {
            assert_eq!(grid[i * n + j], 0.0);
        }
        for &j in &block_b {
            assert!(grid[i * n + j] < -1.0e9, "block A must not see block B");
            assert!(grid[j * n + i] < -1.0e9, "block B must not see block A");
        }
    }
}

/// No image block (pure text) and decode (seq == 1) produce no overlay, so the
/// attention path stays fully causal.
#[test]
#[allow(clippy::unwrap_used, reason = "test fixture arrays always materialise")]
fn overlay_absent_without_image_block() {
    let text_only = [2, 100, 200, 300];
    let seq = text_only.len() as i32;
    let arr = Array::from_i32_slice(&text_only, &[seq]).unwrap();
    assert!(build_vision_bidi_overlay(&arr, seq, Device::Cpu).is_none());

    let one = [2];
    let arr1 = Array::from_i32_slice(&one, &[1]).unwrap();
    assert!(build_vision_bidi_overlay(&arr1, 1, Device::Cpu).is_none());
}
