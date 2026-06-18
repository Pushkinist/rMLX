//! KV-cache sizing for the Qwen3-VL-MoE generate paths.
//!
//! Native Qwen3-VL image tiling produces thousands of image soft tokens (a
//! 2560×2560 image → ~6400 soft tokens), so the augmented prompt routinely
//! exceeds the lazy `KV_MAX_SEQ_DEFAULT = 4096` ring start. The bug these tests
//! pin: the image (and text) generate path built its per-layer caches with the
//! bare 4096 default and never bracketed the prefill, so a >4096-token prompt
//! overflowed the fixed decode buffer with
//! `slice_update: [broadcast_shapes] Shapes (1,4,6776,128) and (1,4,4096,128)`.
//!
//! The fix sizes the ring from the effective `--max-ctx` via
//! [`kv_max_seq_and_ceiling`] and brackets the one-shot prefill with
//! `enter_prefill()` / `exit_prefill()` so the lazy-grow path is used. These
//! tests reproduce that exact cache-construction + prefill pattern at the
//! cache level (CPU, `KvQuant::None`, the model's real head_dim=128 /
//! n_kv_heads=4 shape) without loading the 30B model:
//!
//! * a 6776-token prefill under a 16384 ceiling grows to fit and completes, and
//! * an over-cap prefill is rejected with a clean `KvCeilingExceeded`
//!   (→ HTTP `context_overflow`), not a `slice_update` broadcast panic.

use crate::kv_cache::kv_max_seq_and_ceiling;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

// Real Qwen3-VL-30B-A3B text-decoder KV shape (from the serve log:
// n_kv_heads=4, head_dim=128). max_position_embeddings is large, so a 16384
// `--max-ctx` ceiling resolves to exactly 16384.
const KV_H: i32 = 4;
const HEAD_DIM: i32 = 128;
const QWEN3VL_MPE: i32 = 262_144;

#[allow(
    clippy::expect_used,
    reason = "test helper: .expect() surfaces an Array::from_bytes failure as the test message"
)]
fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md hard rule 1); f32 is 4-byte LE
    // on this target. `data` is borrowed read-only and copied into MLX before
    // the borrow ends.
    #[allow(
        unsafe_code,
        reason = "zero-copy byte view of an f32 slice for Array::from_bytes; copied before the borrow ends"
    )]
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

/// Build the per-layer cache exactly as `generate_image` / `generate_greedy`
/// do, then run a single-shot prefill chunk of `seq` tokens.
fn prefill_one_shot(max_ctx_override: Option<i32>, seq: i32) -> rmlx_core::Result<KvCache> {
    let device = Device::Cpu;
    let (initial_max_seq, max_seq_ceiling) = kv_max_seq_and_ceiling(max_ctx_override, QWEN3VL_MPE);
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, initial_max_seq)
        .with_max_seq_ceiling(max_seq_ceiling)
        .with_layer_idx(0);

    cache.enter_prefill();
    let shape = [1_i32, KV_H, seq, HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.1_f32; n], &shape);
    let v = f32_arr(&vec![0.2_f32; n], &shape);
    cache.update(&k, &v, device)?;
    cache.exit_prefill(device)?;
    Ok(cache)
}

/// A 6776-token image prompt under `--max-ctx 16384` completes: the ring grows
/// past the 4096 default to fit instead of overflowing the fixed decode buffer.
/// This is the headline issue case (large image → ~6400 soft tokens + text).
#[test]
#[allow(
    clippy::expect_used,
    reason = "test asserts the prefill succeeds; .expect() surfaces the failure as the test message"
)]
fn image_prompt_over_4096_fits_under_max_ctx() {
    // 6776 = the issue's augmented length (6715 image soft tokens + text).
    let cache = prefill_one_shot(Some(16_384), 6776)
        .expect("6776-token prefill must fit under a 16384 ceiling");
    assert_eq!(
        cache.offset(),
        6776,
        "offset tracks the full prefilled prompt length",
    );
}

/// Without a ceiling-sized cache (the pre-fix bare 4096 default) the same prompt
/// would overflow; with the ceiling resolved from `--max-ctx` it does not. Pin
/// that the resolved ceiling is the requested value, not the 4096 default.
#[test]
fn max_ctx_override_sizes_ceiling_not_default() {
    let (initial, ceiling) = kv_max_seq_and_ceiling(Some(16_384), QWEN3VL_MPE);
    assert_eq!(
        ceiling, 16_384,
        "ceiling honors --max-ctx, not the 4096 default"
    );
    assert_eq!(
        initial, 4096,
        "ring still starts lazily at the 4096 default and grows up to the ceiling",
    );
}

/// A prompt that exceeds the effective `--max-ctx` is rejected with a clean
/// `KvCeilingExceeded` (mapped to HTTP `context_overflow`), NOT the cryptic
/// `slice_update` broadcast panic the pre-fix path produced.
#[test]
#[allow(
    clippy::panic,
    reason = "test fails loudly if the over-cap prompt is unexpectedly accepted (KvCache has no Debug impl, so expect_err is unavailable)"
)]
fn over_cap_image_prompt_yields_context_overflow_not_broadcast_panic() {
    // ceiling = 4096 (small --max-ctx); a 6776-token prompt is over-cap.
    // KvCache has no Debug impl, so match the Result directly rather than
    // unwrapping the Ok side via expect_err.
    let Err(err) = prefill_one_shot(Some(4096), 6776) else {
        panic!("an over-cap prompt must be rejected before allocation");
    };
    assert!(
        matches!(err, rmlx_core::error::Error::KvCeilingExceeded { .. }),
        "expected KvCeilingExceeded (→ context_overflow), got: {err}",
    );
    let msg = err.to_string();
    assert!(
        msg.contains("exceeds max-ctx ceiling"),
        "error must name the ceiling overflow, not a broadcast shape: {msg}",
    );
    assert!(
        !msg.contains("broadcast"),
        "must NOT surface a raw slice_update broadcast error: {msg}",
    );
}

/// A small image (under 4096 soft tokens) still works — the common control case
/// (e.g. a 448×448 image → 196 soft tokens) must be unaffected by the fix.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test asserts the prefill succeeds; .expect() surfaces the failure as the test message"
)]
fn small_image_prompt_under_default_still_works() {
    let cache =
        prefill_one_shot(Some(16_384), 213).expect("213-token small-image prefill must complete");
    assert_eq!(cache.offset(), 213);
}
