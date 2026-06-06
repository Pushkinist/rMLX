//! Warm-TTFT bf16-K shortcut regression tests for PlanarK.
//!
//! Before this fix, `KvCache::update_planar_k` was the **only** quantised
//! `update_<arch>` that did NOT shortcut to `update_decode_fp16` when the
//! `decode_fp16_k` bf16 seed (set by `exit_prefill`) was live. That
//! asymmetry made PlanarK the sole codec actually exercising 4-bit K
//! reconstruction at decode time, while every other "quantised" KV mode
//! silently reused the bf16 prefill K for the entire decode window. The
//! resulting per-position drift in K compounded across an 8k softmax tail
//! and broke `niah_pflash_bonsai_8k_d50` retrieval.
//!
//! These tests pin the warm-TTFT contract for PlanarK so the regression
//! cannot land again.
//!
//! Test design:
//!
//! * `planar_k_warm_ttft_shortcut_quiescent_codec` — after `enter_prefill`
//!   → chunked-prefill → `exit_prefill`, the bf16 seed is materialised on
//!   `decode_fp16_k`. The first decode step's `update()` MUST route through
//!   the warm-TTFT path and return a K whose accumulated tensor came from
//!   the bf16 buffer (i.e. NOT from a fresh `planar_quantize` round trip).
//!   We assert by checking that the post-decode `QuantPlanarK.shape[2]`
//!   stayed at the pre-decode value (the codec did NOT append).
//! * `planar_k_update_and_sdpa_dispatcher_skips_fused_qk_when_seed_live`
//!   — `update_and_sdpa` MUST route through the legacy fall-through (path
//!   = `"legacy"`) when `decode_fp16_k` is live, NOT through the fused
//!   PlanarK fast path (`path = "planar_k_fused"`).

use super::core::KvCache;
use crate::kvcache::fused_qk_total_dispatch_count;
use crate::planar_flash_decode_msl::planar_flash_decode_dispatch_count;
use crate::storage::{KvStorage, QuantPlanarK};
use crate::test_utils::skip_if_no_gpu_env;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and the bytes are copied
    // into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

/// Bonsai-ish small shape (kv_h=8, head_dim=128, max_seq=512) so the test
/// runs in milliseconds on CPU and GPU alike.
const TEST_KV_H: i32 = 8;
const TEST_HEAD_DIM: i32 = 128;
const TEST_MAX_SEQ: i32 = 512;
const TEST_PREFILL_SEQ: i32 = 256;

/// Drive a PlanarK cache through the full prefill -> decode lifecycle and
/// assert the warm-TTFT shortcut keeps the planar codec quiescent on every
/// decode step.
#[test]
fn planar_k_warm_ttft_shortcut_quiescent_codec() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::PlanarK, TEST_MAX_SEQ);
    assert!(
        matches!(cache.storage(), KvStorage::PlanarK { .. }),
        "PlanarK storage expected"
    );

    cache.enter_prefill();

    // Drive a 256-token prefill through `update()` (one chunk).
    let prefill_shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.123f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.456f32; n_pref], &prefill_shape);
    cache
        .update(&k_pref, &v_pref, device)
        .expect("prefill chunk");

    cache.exit_prefill(device).expect("exit_prefill");

    // After exit_prefill the bf16 K seed must be live, AND PlanarK encoder
    // must have been bulk-loaded with the prefill K codes.
    let pre_decode_codec_seq = planar_k_codec_seq(&cache);
    assert_eq!(
        pre_decode_codec_seq, TEST_PREFILL_SEQ,
        "exit_prefill should bulk-encode prefill K to {TEST_PREFILL_SEQ}, got {pre_decode_codec_seq}"
    );

    // Decode step 1.
    let step_shape = [1i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.789f32; n_step], &step_shape);
    let v_step = f32_arr(&vec![0.321f32; n_step], &step_shape);
    cache
        .update(&k_step, &v_step, device)
        .expect("decode step 1");

    // Warm-TTFT invariant: the shortcut MUST route around the
    // planar encoder. Codec shape[2] should stay frozen.
    let post_decode_codec_seq = planar_k_codec_seq(&cache);
    assert_eq!(
        post_decode_codec_seq, pre_decode_codec_seq,
        "warm-TTFT shortcut violation: PlanarK codec seq advanced from \
         {pre_decode_codec_seq} to {post_decode_codec_seq} on a decode step while \
         `decode_fp16_k` was live (every other quant skips the codec here)"
    );

    // And cache.offset() must reflect the 1 token decode step.
    assert_eq!(
        cache.offset(),
        TEST_PREFILL_SEQ + 1,
        "cache offset after one decode step"
    );
}

/// Helper: peek at `QuantPlanarK.shape[2]` (the codec's accumulated seq
/// length) without exposing it outside the crate.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "test asserts construction-time invariant: cache was created with `KvQuant::PlanarK`, so storage MUST be `KvStorage::PlanarK`. Any other variant is a construction-time bug; an explicit panic catches it sooner than a future variant slipping through."
)]
fn planar_k_codec_seq(cache: &KvCache) -> i32 {
    match cache.storage() {
        KvStorage::PlanarK { k, .. } => k.as_ref().map_or(0, planar_k_shape_seq),
        _ => panic!("expected KvStorage::PlanarK"),
    }
}

fn planar_k_shape_seq(qpk: &QuantPlanarK) -> i32 {
    qpk.shape.get(2).copied().unwrap_or(0)
}

/// Dispatcher gate: `update_and_sdpa` on a warm-TTFT-seeded PlanarK
/// cache MUST NOT dispatch the PlanarK fused-QK or the
/// PlanarFlashDecode kernel. Both counters must stay flat.
///
/// This is a GPU test because the `update_and_sdpa_planar_k_fused` gate
/// only fires when `device == Device::Gpu`. Mark `#[ignore]` per project
/// policy; run via:
///   cargo test -p rmlx-kv-quant planar_k_update_and_sdpa_dispatcher_skips_fused_qk_when_seed_live \
///     -- --ignored --test-threads=1
#[test]
#[ignore = "GPU Metal context — run with --ignored --test-threads=1"]
fn planar_k_update_and_sdpa_dispatcher_skips_fused_qk_when_seed_live() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::PlanarK, TEST_MAX_SEQ);

    cache.enter_prefill();

    let prefill_shape = [1i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.123f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.456f32; n_pref], &prefill_shape);
    cache
        .update(&k_pref, &v_pref, device)
        .expect("prefill chunk");
    cache.exit_prefill(device).expect("exit_prefill");

    // Snapshot both dispatch counters before the decode step.
    let pflash_before = planar_flash_decode_dispatch_count();
    let fused_qk_before = fused_qk_total_dispatch_count();

    // Decode step (q_seq == 1, the shape that would trigger both fused-QK and
    // flash-decode on an unseeded cache).
    let step_shape = [1i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let q_shape = [1i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.789f32; n_step], &step_shape);
    let v_step = f32_arr(&vec![0.321f32; n_step], &step_shape);
    let q_step = f32_arr(&vec![0.111f32; n_step], &q_shape);

    let scale = 1.0f32 / (TEST_HEAD_DIM as f32).sqrt();
    cache
        .update_and_sdpa(&q_step, &k_step, &v_step, scale, "causal", None, device)
        .expect("update_and_sdpa decode step 1");

    let pflash_after = planar_flash_decode_dispatch_count();
    let fused_qk_after = fused_qk_total_dispatch_count();

    // Both kernels MUST stay dormant when the bf16 K seed is live.
    assert_eq!(
        pflash_after - pflash_before,
        0,
        "dispatcher gate: PlanarFlashDecode kernel MUST stay dormant \
         when decode_fp16_k seed is live (warm-TTFT); delta={}",
        pflash_after - pflash_before,
    );
    assert_eq!(
        fused_qk_after - fused_qk_before,
        0,
        "sanity: fused-QK kernel (K8V*/TurboSym*) MUST NOT fire on a \
         PlanarK cache (no kernel registered for PlanarK in the fused-QK \
         table); delta={}",
        fused_qk_after - fused_qk_before,
    );
}
