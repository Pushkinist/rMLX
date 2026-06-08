//! Windowed-layer (SWA) KV-ring sizing diagnostic — issue #35.
//!
//! Issue #35 alleges that KV for sliding-window-attention (SWA / windowed)
//! layers is allocated to the FULL context length rather than the attention
//! window, so windowed-layer KV memory scales with context. These tests are the
//! falsification harness: they drive a windowed (`rotating`) cache and a global
//! (full-attention) cache through prefills at growing context (4k / 16k / 64k)
//! and read [`KvCache::resident_bytes`] — the actual on-device byte accounting
//! added in issue #33 — to settle the claim.
//!
//! All tests run on `Device::Cpu` (no GPU / no model snapshot required); the
//! rotating ring path is device-agnostic, so the byte arithmetic is identical
//! to the on-device Gemma4 SWA path. Real-model proof at 64k is a separate
//! later step.
//!
//! Verdict pinned by these tests:
//!
//! 1. `windowed_ring_stays_bounded_while_global_grows` — at 4k/16k/64k the
//!    windowed ring's `resident_bytes` is FLAT at `window + chunk` per side,
//!    while the global cache grows linearly with context (64× larger at 64k).
//!    This directly falsifies the #35 claim for the rotating SWA path.
//! 2. `windowed_ring_retains_full_swa_window` — correctness guard: across a
//!    chunked prefill longer than the window AND multiple decode steps, the
//!    physical ring is bounded to `[window, window + chunk]` rows (never the
//!    full context), and the retained token set is always a superset of the
//!    most-recent `window` tokens — the full SWA-attended set. No still-attended
//!    key is ever evicted. (`offset` tracks the logical position, mirroring
//!    mlx-lm `RotatingKVCache.offset`, not the physical ring fill.)

use super::core::KvCache;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

// SWA layer shape for Gemma4 e2b/e4b: kv_h=1 baseline; kv_h=2 tested by
// `windowed_ring_flat_across_ctx_kv_h2` (confirms the "linear in kv_h" claim).
// head_dim=256 is the Gemma4 SWA head_dim.
const B: i32 = 1;
const KV_H: i32 = 1;
const D: i32 = 256;
/// Gemma4 e2b/e4b sliding window.
const WINDOW: i32 = 512;

/// Build a `[B, kv_h, seq, D]` bf16 Array filled with zeros on CPU.
#[allow(
    clippy::unwrap_used,
    reason = "test helper — panic on failure is the desired outcome"
)]
fn bf16_zeros(seq: i32) -> Array {
    let n = (B * KV_H * seq * D) as usize;
    let bytes = vec![0u8; n * 2];
    Array::from_bytes(&bytes, &[B, KV_H, seq, D], Dtype::Bf16).unwrap()
}

/// Truncate an `f32` to bf16 (drop the low 16 mantissa bits) and return the
/// two big-endian-free little-endian bytes MLX stores. Round-trip with
/// [`bf16_bytes_to_f32`] is exact, so token identity survives the ring.
fn f32_to_bf16_bytes(x: f32) -> [u8; 2] {
    let hi = (x.to_bits() >> 16) as u16;
    hi.to_le_bytes()
}

/// Inverse of [`f32_to_bf16_bytes`]: widen a stored bf16 back to f32.
fn bf16_bytes_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = u32::from(u16::from_le_bytes([lo, hi])) << 16;
    f32::from_bits(bits)
}

/// Build a `[B, kv_h, seq, D]` bf16 Array where every element in token row `t`
/// equals `base + t` (truncated to bf16). Lets a later test assert *which*
/// tokens the ring retained, not just how many. The correctness test keeps
/// `base + t < 256` so every value is an exact, distinct bf16 (no aliasing) and
/// the retained-set assertions are crisp.
#[allow(
    clippy::unwrap_used,
    reason = "test helper — panic on failure is the desired outcome"
)]
fn bf16_ramp(seq: i32, base: i32) -> Array {
    let row_elems = (B * KV_H * D) as usize;
    let mut bytes = Vec::with_capacity((seq as usize) * row_elems * 2);
    for t in 0..seq {
        let b2 = f32_to_bf16_bytes((base + t) as f32);
        for _ in 0..row_elems {
            bytes.extend_from_slice(&b2);
        }
    }
    Array::from_bytes(&bytes, &[B, KV_H, seq, D], Dtype::Bf16).unwrap()
}

/// Real Gemma4 prefill chunk size — SWA trimming happens per chunk, not in one
/// shot (mlx-lm RotatingKVCache only trims on the SECOND+ `update_and_fetch`).
const PREFILL_CHUNK: i32 = 512;

/// Resident bytes of a WINDOWED (rotating) cache after prefilling `ctx` tokens.
/// Mirrors the Gemma4 SWA-layer construction:
/// `KvCache::with_quant_max_seq_window(quant, initial_max_seq, Some(window))`.
///
/// Uses the module-level `KV_H=1` shape.
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn windowed_bytes_after_prefill(ctx: i32) -> u64 {
    windowed_bytes_after_prefill_kv_h(ctx, KV_H)
}

/// Parameterised variant of [`windowed_bytes_after_prefill`] that lets the
/// caller vary `kv_h` independently of the module-level constant. Used to
/// assert that the flat-across-ctx property holds for `kv_h > 1` (linearity
/// by construction: `resident_bytes` counts the shape product `B×kv_h×rows×D`).
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn windowed_bytes_after_prefill_kv_h(ctx: i32, kv_h: i32) -> u64 {
    let device = Device::Cpu;
    // initial_max_seq is the lazy default; the rotating path ignores it and
    // sizes to `window`. quant is recorded but unused on the rotating path
    // (SWA stays bf16 regardless — mlx-lm RotatingKVCache.to_quantized raises).
    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::K8V8, 4096, Some(WINDOW));
    // Drive chunked prefill using arrays shaped [B, kv_h, chunk, D].
    cache.enter_prefill();
    let mut pos = 0;
    while pos < ctx {
        let s = PREFILL_CHUNK.min(ctx - pos);
        let n = (B * kv_h * s * D) as usize;
        let bytes = vec![0u8; n * 2];
        let k = Array::from_bytes(&bytes, &[B, kv_h, s, D], Dtype::Bf16).unwrap();
        let v = Array::from_bytes(&bytes, &[B, kv_h, s, D], Dtype::Bf16).unwrap();
        cache.update(&k, &v, device).unwrap();
        pos += s;
    }
    cache.exit_prefill(device).unwrap();
    cache.resident_bytes()
}

/// Resident bytes of a GLOBAL (full-attention) cache after prefilling `ctx`
/// tokens. bf16 path stores the compact fp16 decode seed sized to `ctx`.
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn global_bytes_after_prefill(ctx: i32) -> u64 {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, ctx);
    let k = bf16_zeros(ctx);
    let v = bf16_zeros(ctx);
    cache.enter_prefill();
    cache.update(&k, &v, device).unwrap();
    cache.exit_prefill(device).unwrap();
    cache.resident_bytes()
}

/// #35 falsification: windowed-layer ring bytes are FLAT across context while
/// the global-layer cache grows linearly. If the ticket were correct the
/// windowed bytes would track the global bytes (both ∝ ctx).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn windowed_ring_stays_bounded_while_global_grows() {
    // One bf16 key OR value token row = B × kv_h × D × 2 bytes.
    let bytes_per_token_row = (B * KV_H * D) as u64 * 2;
    // Ring holds K + V. During chunked `_update_concat` prefill the rotating
    // buffer is transiently `max_size - 1 + chunk` rows (mlx-lm RotatingKVCache
    // trims to `max_size - 1` then concatenates the new chunk), so the physical
    // cap is `window + chunk` per side — bounded by the WINDOW + one CHUNK,
    // NEVER by context length. That bound (not exactly `window`) is the
    // model-agnostic invariant.
    let window_cap_tokens = (WINDOW + PREFILL_CHUNK) as u64;
    let window_cap_bytes = 2 * window_cap_tokens * bytes_per_token_row;

    let mut prev_global = 0u64;
    for &ctx in &[4096_i32, 16_384, 65_536] {
        let w = windowed_bytes_after_prefill(ctx);
        let g = global_bytes_after_prefill(ctx);
        eprintln!(
            "[#35] ctx={ctx:>6}  windowed={w:>12}  global={g:>12}  \
             window_cap={window_cap_bytes}"
        );

        // Windowed ring never exceeds the window capacity (K+V × window tokens),
        // regardless of context length. This is the core #35 claim, falsified.
        assert!(
            w <= window_cap_bytes,
            "windowed ring bytes ({w}) must stay <= window cap ({window_cap_bytes}) at ctx={ctx} \
             — over-allocation would confirm #35"
        );

        // Global cache grows strictly with context (sanity: the harness is
        // actually scaling context, so the flat windowed result is meaningful).
        assert!(
            g > prev_global,
            "global cache bytes ({g}) must grow with context (prev={prev_global}) at ctx={ctx}"
        );
        prev_global = g;
    }

    // At 64k the windowed ring is dramatically smaller than the global cache:
    // window/ctx = 512/65536 = 1/128. Assert at least a 50× gap to lock in that
    // the windowed layer does NOT scale with context.
    let w64 = windowed_bytes_after_prefill(65_536);
    let g64 = global_bytes_after_prefill(65_536);
    assert!(
        g64 >= w64 * 50,
        "at 64k the global cache ({g64}) must be >=50x the windowed ring ({w64}) — \
         confirms windowed KV does not scale with context"
    );

    // Flatness / equality pin (docs/KV_CACHE.md §4.6).
    //
    // docs/KV_CACHE.md documents the windowed ring as FLAT at exactly
    // 1,047,552 bytes for kv_h=1, head_dim=256, window=512.  Pin the live
    // measurement against that value so a silent bloat (still under
    // `window_cap_bytes`) can't escape this test.
    let w4 = windowed_bytes_after_prefill(4_096);
    assert_eq!(
        w4, 1_047_552,
        "windowed ring at 4k must equal the documented flat bound \
         1,047,552 B (docs/KV_CACHE.md §4.6)"
    );
    assert_eq!(
        windowed_bytes_after_prefill(16_384),
        w4,
        "windowed ring must be flat (identical bytes) across ctx 4k→16k \
         (docs/KV_CACHE.md §4.6)"
    );
    assert_eq!(
        windowed_bytes_after_prefill(65_536),
        w4,
        "windowed ring must be flat (identical bytes) across ctx 4k→64k \
         (docs/KV_CACHE.md §4.6)"
    );
}

/// Verify the flat-across-ctx property holds for `kv_h=2` (the documented
/// "linear in kv_h" claim, now a tested fact).
///
/// `resident_bytes` counts the full shape product `B×kv_h×rows×D×2`, so
/// doubling `kv_h` must exactly double the flat bound, and the result must
/// still be perfectly flat across 4k/16k/64k.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test — panics are the intended failure mode"
)]
fn windowed_ring_flat_across_ctx_kv_h2() {
    let w4 = windowed_bytes_after_prefill_kv_h(4_096, 2);
    // kv_h=2 doubles every element count → exactly 2× the kv_h=1 flat bound.
    assert_eq!(
        w4,
        1_047_552 * 2,
        "kv_h=2 windowed ring at 4k must equal 2× the kv_h=1 flat bound \
         (linearity: resident_bytes ∝ kv_h)"
    );
    assert_eq!(
        windowed_bytes_after_prefill_kv_h(16_384, 2),
        w4,
        "kv_h=2 windowed ring must be flat across ctx 4k→16k"
    );
    assert_eq!(
        windowed_bytes_after_prefill_kv_h(65_536, 2),
        w4,
        "kv_h=2 windowed ring must be flat across ctx 4k→64k"
    );
}

/// Correctness guard: through a chunked prefill longer than the window AND
/// several decode steps, the rotating ring always retains the FULL most-recent
/// `window` tokens — the entire SWA-attended set — so no still-attended key is
/// ever evicted. The physical buffer never grows to the full context.
///
/// Uses a small exact-bf16 scale (`window=8`, `chunk=4`, all token values < 256
/// so each is a distinct, exactly-representable bf16) to make token identity
/// crisp and the retained-set assertions exact — no bf16 aliasing. This
/// exercises BOTH the multi-chunk `_update_concat` prefill path and the
/// per-token `_update_in_place` decode path (the real per-step eviction risk).
#[test]
#[allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test — panics are the intended failure mode; shapes fixed by construction"
)]
fn windowed_ring_retains_full_swa_window() {
    const W: i32 = 8; // small exact-bf16 window
    const CH: i32 = 4; // small chunk
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::None, 64, Some(W));

    // Exact-bf16 identity helpers (values < 256 are exact in bf16).
    let trunc_bits = |x: i32| -> u32 {
        let b = f32_to_bf16_bytes(x as f32);
        bf16_bytes_to_f32(b[0], b[1]).to_bits()
    };
    let read_rows = |arr: &Array| -> std::collections::HashSet<u32> {
        arr.eval().unwrap();
        let bytes = arr.to_bytes().unwrap();
        let row_elems = (B * KV_H * D) as usize;
        let rows = arr.shape()[2] as usize;
        (0..rows)
            .map(|r| {
                let off = r * row_elems * 2;
                bf16_bytes_to_f32(bytes[off], bytes[off + 1]).to_bits()
            })
            .collect()
    };
    // Assert: retained ⊇ the most-recent `W` tokens of `head` (SWA-attended set),
    // and retained ⊆ the most-recent `W + CH` tokens (never the full context).
    let assert_window_retained = |retained: &std::collections::HashSet<u32>, head: i32| {
        for tok in (head - W).max(0)..head {
            assert!(
                retained.contains(&trunc_bits(tok)),
                "still-attended token {tok} (within window of head {head}) must be retained — \
                 eviction here would silently corrupt SWA output"
            );
        }
        let allowed: std::collections::HashSet<u32> =
            ((head - (W + CH)).max(0)..head).map(trunc_bits).collect();
        assert!(
            retained.is_subset(&allowed),
            "no token older than `window+chunk` back may survive (head={head})"
        );
    };

    // --- chunked prefill of 40 tokens (5 chunks of 4 → trims past the window) ---
    let prefill_len = 40_i32;
    cache.enter_prefill();
    let mut pos = 0;
    let mut last_k: Option<Array> = None;
    while pos < prefill_len {
        let s = CH.min(prefill_len - pos);
        let (kk, _vv) = cache
            .update(&bf16_ramp(s, pos), &bf16_ramp(s, pos), device)
            .unwrap();
        last_k = Some(kk);
        pos += s;
    }
    cache.exit_prefill(device).unwrap();

    assert_eq!(
        cache.offset(),
        prefill_len,
        "rotating offset tracks logical position (total tokens seen)"
    );
    let k_out = last_k.unwrap();
    let buf_rows = k_out.shape()[2];
    assert!(
        (W..=W + CH).contains(&buf_rows),
        "ring physical rows ({buf_rows}) must be in [window, window+chunk] — \
         never the full {prefill_len}-token context"
    );
    assert_window_retained(&read_rows(&k_out), prefill_len);

    // --- decode steps: single-token in-place updates (the per-step eviction
    //     risk). After each step the most-recent `W` tokens must still be live. ---
    for step in 0..12_i32 {
        let head = prefill_len + step; // value of the new token
        let (kk, _vv) = cache
            .update(&bf16_ramp(1, head), &bf16_ramp(1, head), device)
            .unwrap();
        // logical head after appending this token is head+1.
        assert_window_retained(&read_rows(&kk), head + 1);
    }
}
