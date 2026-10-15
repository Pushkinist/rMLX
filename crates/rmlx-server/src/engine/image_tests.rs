//! Image-prompt splice-placement tests.
//!
//! These cover [`splice_image_block`] — the token-level placement of the
//! per-image `<boi> + N×image + <eoi>` block. The end-to-end vision decode is
//! exercised by the real-model integration path; here we assert the block lands
//! *inside* the user turn (the bug this logic fixes was inserting it before the
//! turn, leaving the image outside the question).
#![allow(
    clippy::indexing_slicing,
    reason = "test fixtures: index/slice bounds are fixed by the literal prompt arrays constructed in each test"
)]

use super::{
    preprocess_failure, resolve_gemma4_processor, rmlx_server_load_image, splice_image_block,
    GEMMA4_USER_TURN_OPENER, QWEN3VL_USER_TURN_OPENER,
};
use rmlx_core::Error;

/// A minimal image block placeholder (real blocks are `<boi> + N×img + <eoi>`;
/// the placement logic is independent of the block contents).
fn block(marker: u32, n: usize) -> Vec<u32> {
    let mut b = vec![900];
    b.extend(std::iter::repeat_n(marker, n));
    b.push(901);
    b
}

#[test]
fn gemma4_block_inserts_inside_user_turn() {
    // <bos> <|turn> user \n  "hi"  <turn|> \n
    let bos = 2u32;
    let opener = GEMMA4_USER_TURN_OPENER; // [105, 2364, 107]
    let prompt = [bos, opener[0], opener[1], opener[2], 555, 106, 107];
    let blk = block(258_880, 3);
    let aug = splice_image_block(&prompt, std::slice::from_ref(&blk), &opener);

    // The block must appear AFTER the opener and BEFORE the user text token 555.
    let opener_end = 4; // bos + 3 opener tokens
    assert_eq!(
        &aug[..opener_end],
        &prompt[..opener_end],
        "prefix preserved"
    );
    assert_eq!(
        &aug[opener_end..opener_end + blk.len()],
        blk.as_slice(),
        "block spliced right after the user-turn opener"
    );
    assert_eq!(
        aug[opener_end + blk.len()],
        555,
        "user text follows the block (image is inside the turn, before the text)"
    );
}

#[test]
fn gemma4_block_uses_last_user_turn_in_multi_turn() {
    let bos = 2u32;
    let o = GEMMA4_USER_TURN_OPENER;
    // two user turns; image accompanies the second (last) one.
    let prompt = [
        bos, o[0], o[1], o[2], 10, 106, 107, // turn 1: user "10"
        o[0], o[1], o[2], 20, 106, 107, // turn 2: user "20"
    ];
    let blk = block(258_880, 2);
    let aug = splice_image_block(&prompt, std::slice::from_ref(&blk), &o);

    // Insert position is after the SECOND opener (index 7 + 3 = 10).
    let second_opener_end = 10;
    assert_eq!(
        &aug[second_opener_end..second_opener_end + blk.len()],
        blk.as_slice(),
        "block spliced into the final user turn, not the first"
    );
    assert_eq!(
        aug[second_opener_end + blk.len()],
        20,
        "second turn text follows"
    );
    // The first turn must be untouched (no block before token 10).
    assert_eq!(&aug[..second_opener_end], &prompt[..second_opener_end]);
}

#[test]
fn falls_back_to_after_bos_when_opener_absent() {
    let bos = 2u32;
    let prompt = [bos, 500, 501, 502];
    let blk = block(258_880, 2);
    // Empty opener forces the after-BOS fallback (e.g. the Gemma3 path).
    let aug = splice_image_block(&prompt, std::slice::from_ref(&blk), &[]);
    assert_eq!(aug[0], bos, "BOS stays first");
    assert_eq!(
        &aug[1..=blk.len()],
        blk.as_slice(),
        "block spliced right after BOS"
    );
    assert_eq!(
        &aug[1 + blk.len()..],
        &prompt[1..],
        "rest of prompt preserved"
    );

    // A non-empty opener that is not present also falls back to after-BOS.
    let aug2 = splice_image_block(
        &prompt,
        std::slice::from_ref(&blk),
        &GEMMA4_USER_TURN_OPENER,
    );
    assert_eq!(
        &aug2[1..=blk.len()],
        blk.as_slice(),
        "opener absent → after-BOS"
    );
}

#[test]
fn empty_prompt_inserts_block_at_front() {
    let blk = block(258_880, 1);
    let aug = splice_image_block(&[], std::slice::from_ref(&blk), &GEMMA4_USER_TURN_OPENER);
    assert_eq!(aug, blk, "empty prompt → block is the whole sequence");
}

#[test]
fn multi_image_blocks_appear_in_order_and_contiguous() {
    // Two blocks passed together must land in-order and contiguous at the splice
    // point — no interleaving, no reversal.
    let bos = 2u32;
    let o = GEMMA4_USER_TURN_OPENER;
    let prompt = [bos, o[0], o[1], o[2], 555, 106, 107];
    let blk1 = block(111, 2); // first image block
    let blk2 = block(222, 3); // second image block
    let aug = splice_image_block(&prompt, &[blk1.clone(), blk2.clone()], &o);

    let opener_end = 4usize; // bos + 3 opener tokens
                             // Both blocks appear right after the opener, blk1 first, blk2 immediately after.
    assert_eq!(
        &aug[opener_end..opener_end + blk1.len()],
        blk1.as_slice(),
        "first block lands first"
    );
    assert_eq!(
        &aug[opener_end + blk1.len()..opener_end + blk1.len() + blk2.len()],
        blk2.as_slice(),
        "second block immediately follows first"
    );
    // User text token 555 follows both blocks without any prompt tokens in between.
    assert_eq!(
        aug[opener_end + blk1.len() + blk2.len()],
        555,
        "user text follows both blocks"
    );
    // Total length is consistent.
    assert_eq!(
        aug.len(),
        prompt.len() + blk1.len() + blk2.len(),
        "length adds up"
    );
}

// ---- image-token budget override resolution ---------------------------

#[test]
fn resolve_processor_none_borrows_shared() {
    let proc = rmlx_models::gemma4::Gemma4ImageProcessor::with_defaults();
    let resolved = resolve_gemma4_processor(&proc, None);
    // No override → borrow (zero alloc), budget unchanged.
    assert!(matches!(resolved, std::borrow::Cow::Borrowed(_)));
    assert_eq!(resolved.config().max_soft_tokens, 280);
}

#[test]
fn resolve_processor_override_owns_with_clamped_budget() {
    let proc = rmlx_models::gemma4::Gemma4ImageProcessor::with_defaults();

    let raised = resolve_gemma4_processor(&proc, Some(560));
    assert!(matches!(raised, std::borrow::Cow::Owned(_)));
    assert_eq!(raised.config().max_soft_tokens, 560);

    // Above the safe ceiling → clamped to the model bound.
    let clamped = resolve_gemma4_processor(&proc, Some(999_999));
    assert_eq!(
        clamped.config().max_soft_tokens,
        rmlx_models::gemma4::MAX_SUPPORTED_SOFT_TOKENS
    );

    // Shared processor is never mutated.
    assert_eq!(proc.config().max_soft_tokens, 280);
}

#[test]
fn qwen3vl_single_turn_splice_matches_last_opener() {
    // Verify that routing Qwen3-VL through splice_image_block (last-match)
    // produces the same insert position as the former first-match code for a
    // single-turn prompt (first == last when there is only one opener).
    let bos = 1u32;
    let o = QWEN3VL_USER_TURN_OPENER; // [151644, 872, 198]
    let prompt = [bos, o[0], o[1], o[2], 999, 151_645];
    let blk = block(151_655, 4);
    let aug = splice_image_block(&prompt, std::slice::from_ref(&blk), &o);

    let opener_end = 4usize; // bos + 3 opener tokens
    assert_eq!(
        &aug[opener_end..opener_end + blk.len()],
        blk.as_slice(),
        "vision block lands right after the user-turn opener"
    );
    assert_eq!(aug[opener_end + blk.len()], 999, "user text follows");
}

/// A source the server cannot read (bad base64, a missing file) is the
/// client's input: `InvalidInput`, which names the fact.
#[test]
fn an_image_source_that_cannot_be_loaded_is_invalid_input() {
    let bad_base64 = rmlx_server_load_image("!!not base64!!").expect_err("bad base64");
    assert!(
        matches!(bad_base64, Error::InvalidInput(_)),
        "{bad_base64:?}"
    );
    assert!(
        bad_base64.to_string().contains("image load failed"),
        "{bad_base64}"
    );

    let home_file = rmlx_server_load_image("~/rmlx-no-such-dir/image.png").expect_err("no file");
    assert!(matches!(home_file, Error::InvalidInput(_)), "{home_file:?}");
    assert_eq!(
        home_file.to_string(),
        "image load failed: cannot read image file ~/rmlx-no-such-dir/image.png"
    );
    assert!(!home_file.is_migratable());
}

/// Bytes that load but are not an image the processor takes are the client's:
/// `InvalidInput`, which names the image.
#[test]
fn an_image_the_processor_refuses_is_invalid_input() {
    let processor = rmlx_models::gemma4::Gemma4ImageProcessor::with_defaults();
    let err = processor
        .preprocess(b"not an image")
        .map_err(|e| preprocess_failure(2, &e))
        .expect_err("not an image");
    assert!(matches!(err, Error::InvalidInput(_)), "{err:?}");
    assert!(
        err.to_string().contains("image 2 preprocess failed"),
        "{err}"
    );
    assert!(
        !err.to_string().contains("model:") && !err.to_string().contains("gemma4"),
        "the text holds an error class or architecture prefix: {err}"
    );
    assert!(!err.is_migratable());
}

/// Every call of an image processor in `image.rs` maps its error through
/// `preprocess_failure`, so none of them leaves the client's bad image as a
/// retryable `Other`. The four calls are the three Gemma loops and Qwen3-VL.
#[test]
fn every_image_processor_call_maps_its_error_through_preprocess_failure() {
    let code: String = include_str!("image.rs")
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    let statements: Vec<&str> = [".preprocess(", "qwen3_vl_moe::preprocess("]
        .iter()
        .flat_map(|call| code.match_indices(call).map(|(at, _)| at))
        .filter_map(|at| code.get(at..)?.split(';').next())
        .collect();
    assert_eq!(statements.len(), 4, "{statements:?}");
    for statement in statements {
        assert!(
            statement.contains(".map_err(|e| preprocess_failure("),
            "an image processor call does not map its error through preprocess_failure: {statement}"
        );
    }
}
