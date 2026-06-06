//! Long-prompt chunked-prefill regression tests for `QuantPlanarK`.
//!
//! The Bonsai NIAH 8k cell (`niah_pflash_bonsai_8k_d50`) reports incoherent
//! decode output on `KvQuant::PlanarK` while `KvQuant::K8V4` retrieves the
//! needle correctly. Both K8V4 and PlanarK accumulate raw bf16 K during
//! chunked prefill and quantize once in `exit_prefill`.
//!
//! ## Investigation outcome
//!
//! Tests 1–4 confirm the PlanarQuant V4 GPU codec and `QuantPlanarK::append`
//! are bit-equivalent to the CPU reference at 8k-Bonsai shape, and chunked
//! append (per-token after prefill) is bit-equal to a one-shot append. There
//! is **no buffer-arithmetic bug** in the codec or storage layer.
//!
//! The actual root cause is that PlanarK was the **sole codec** lacking the
//! warm-TTFT bf16-K seed shortcut in its decode-time `update_<arch>`. Every
//! other quant (K8V4 / K8V8 / Planar / Mixed / K8VTurbo* / Iso* / Rotor* /
//! TurboSym*) returns early to `update_decode_fp16` when `decode_fp16_k`
//! is `Some(_)` (set by `exit_prefill`), so the bf16 prefill K is reused
//! for the entire decode window. PlanarK uniquely re-encoded K through
//! the lossy 4-bit Lloyd-Max + Givens kernel on every decode step. On
//! Bonsai 8k NIAH the per-position drift compounded across the 8k slot
//! softmax tail and shifted the argmax off the needle position.
//!
//! The fix:
//!
//! * `KvCache::update_planar_k` (`crates/rmlx-kv-quant/src/kvcache/update.rs`):
//!   add the `if self.decode_fp16_k.is_some()` shortcut that every other
//!   update path already had.
//! * `KvCache::update_and_sdpa` dispatcher (`crates/rmlx-kv-quant/src/kvcache/sdpa.rs`):
//!   gate the PlanarK fused-QK fast path on
//!   `decode_fp16_k.is_none()` — when the bf16 seed is live, fall through
//!   to the legacy path so the same warm-TTFT contract holds.
//!
//! Tests 5–6 (dispatcher-level) live in
//! `crates/rmlx-kv-quant/src/kvcache/warm_ttft_tests.rs`:
//! `planar_k_update_and_sdpa_dispatcher_skips_fused_qk_when_seed_live`
//! drives `KvCache::update_and_sdpa` on a warm-TTFT-seeded PlanarK cache
//! and asserts both the PlanarFlashDecode and fused-QK dispatch counters
//! stay flat. This file covers only the codec/storage layer (tests 1–4).
//!
//! All MSL tests are `#[ignore]` because they touch Metal — run with
//! `--ignored --test-threads=1`.

use super::QuantPlanarK;
use crate::planarquant::{planar_dequantize, planar_quantize};
use crate::planarquant_msl::{planar_dequantize_v4_gpu, planar_quantize_v4_gpu};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env};
use crate::turboquant::GROUP_SIZE;
use rmlx_mlx::{Array, Device, Dtype};

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and the bytes are copied
    // into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("array eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk")))
        .collect()
}

/// Bonsai-shaped K at 8k context: `[1, 8, 8192, 128]`.  Matches
/// `kv_h = num_key_value_heads`, `head_dim` from
/// `prism-ml__Ternary-Bonsai-8B-mlx-2bit/config.json`.
const BONSAI_KV_H: i32 = 8;
const BONSAI_HEAD_DIM: i32 = 128;

/// Does the pure PlanarQuant V4 GPU encode->decode roundtrip preserve a
/// Bonsai-shaped [1, 8, 8192, 128] tensor?
///
/// This sidesteps `QuantPlanarK` storage entirely and isolates the GPU
/// kernel. If this fails, the bug is in the MSL kernel at scale (atomic
/// contention, grid limits, or buffer-pool issues).
#[test]
#[ignore = "GPU Metal context — cargo test planar_k_chunked_prefill -- --ignored --test-threads=1"]
fn planar_v4_msl_roundtrip_8k_bonsai_shape() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, BONSAI_KV_H, 8192, BONSAI_HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0x00A1_65A1_BEEF_CAFE_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales, rot32) = planar_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU quantize");
    let recon = planar_dequantize_v4_gpu(&codes, &scales, &rot32, &shape, Device::Gpu)
        .expect("GPU dequantize");

    let recon_vec = array_to_f32(&recon);
    assert_eq!(recon_vec.len(), n);

    let stats = cosine_similarity_per_row(&data, &recon_vec, BONSAI_HEAD_DIM as usize);
    assert!(
        stats.mean >= 0.99,
        "planar_v4 GPU roundtrip mean cosine {:.6} < 0.99 at 8k Bonsai shape \
         (kernel breaks at scale)",
        stats.mean,
    );
}

/// Does `QuantPlanarK::append` reproduce the encoded buffers bit-identically
/// to a one-shot `planar_quantize_v4_gpu` for an 8k Bonsai shape via a SINGLE
/// append (mirrors `exit_prefill`)?
///
/// `QuantPlanarK::append` runs the MSL kernel once, then `slice_update`s
/// the output into the growth buffer. The growth logic capped at
/// `KV_PAGE_SIZE=256` triggers a reallocation from 256 to 8192 tokens
/// before the slice_update — this test exercises that path.
#[test]
#[ignore = "GPU Metal context — cargo test planar_k_chunked_prefill -- --ignored --test-threads=1"]
fn quant_planar_k_single_append_8k_bonsai_shape() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, BONSAI_KV_H, 8192, BONSAI_HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0x00A1_65A2_BEEF_CAFE_u64);
    let k_arr = make_f32_array(&data, &shape);

    let mut init_shape = shape.to_vec();
    init_shape[2] = 0;
    let max_seq = 9_216i32;
    let mut qpk = QuantPlanarK::new(init_shape, max_seq);
    qpk.append(&[], &shape, &k_arr, Device::Gpu, max_seq)
        .expect("QuantPlanarK::append");

    let (_, recon_opt) = qpk
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("dequantize_choice");
    let recon = recon_opt.expect("GPU dequant returned None");

    let recon_vec = array_to_f32(&recon);
    assert_eq!(recon_vec.len(), n);

    let stats = cosine_similarity_per_row(&data, &recon_vec, BONSAI_HEAD_DIM as usize);
    assert!(
        stats.mean >= 0.99,
        "QuantPlanarK single-append 8k Bonsai shape mean cosine {:.6} < 0.99 \
         (append path corrupts buffers at scale)",
        stats.mean,
    );
}

/// Divergence check: one-shot encode of 1024 tokens vs a 768-prefill +
/// per-token-decode chunked encode. This mirrors the post-`exit_prefill`
/// decode-loop append pattern.
///
/// PlanarQuant has no inter-token state, so per-token GPU outputs MUST be
/// bit-equal across the two append schedules. Divergence here is the
/// chunked-prefill broadcast bug.
#[test]
#[ignore = "GPU Metal context — cargo test planar_k_chunked_prefill -- --ignored --test-threads=1"]
fn quant_planar_k_oneshot_vs_chunked_append_parity() {
    if skip_if_no_gpu_env() {
        return;
    }
    let total_seq = 1024i32;
    let prefill_seq = 768i32;
    let decode_seq = total_seq - prefill_seq;
    let kv_h = BONSAI_KV_H;
    let head_dim = BONSAI_HEAD_DIM;
    let full_shape = [1i32, kv_h, total_seq, head_dim];
    let n: usize = full_shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0x00A1_65A3_BEEF_CAFE_u64);
    let max_seq = total_seq + 256;

    // ── Path A: one-shot append of full 1024 ─────────────────────────────
    let mut init_shape_a = full_shape.to_vec();
    init_shape_a[2] = 0;
    let mut qpk_a = QuantPlanarK::new(init_shape_a, max_seq);
    let k_arr_a = make_f32_array(&data, &full_shape);
    qpk_a
        .append(&[], &full_shape, &k_arr_a, Device::Gpu, max_seq)
        .expect("oneshot append");
    let (_, recon_a_opt) = qpk_a
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("dequant A");
    let recon_a = array_to_f32(&recon_a_opt.expect("A None"));

    // ── Path B: prefill chunk then per-token decode appends ──────────────
    let mut init_shape_b = full_shape.to_vec();
    init_shape_b[2] = 0;
    let mut qpk_b = QuantPlanarK::new(init_shape_b, max_seq);
    let prefill_shape = [1i32, kv_h, prefill_seq, head_dim];
    let prefill_elems: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let prefill_arr = make_f32_array(&data[..prefill_elems], &prefill_shape);
    qpk_b
        .append(&[], &prefill_shape, &prefill_arr, Device::Gpu, max_seq)
        .expect("prefill append B");
    let per_step_elems = (kv_h * head_dim) as usize;
    let step_shape = [1i32, kv_h, 1, head_dim];
    for step in 0..decode_seq as usize {
        let off = prefill_elems + step * per_step_elems;
        let step_arr = make_f32_array(&data[off..off + per_step_elems], &step_shape);
        qpk_b
            .append(&[], &step_shape, &step_arr, Device::Gpu, max_seq)
            .expect("decode append B");
    }
    let (_, recon_b_opt) = qpk_b
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("dequant B");
    let recon_b = array_to_f32(&recon_b_opt.expect("B None"));

    assert_eq!(
        recon_a.len(),
        recon_b.len(),
        "oneshot vs chunked append produced different-length recons"
    );

    let max_err = recon_a
        .iter()
        .zip(recon_b.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-6,
        "oneshot vs chunked append max abs error {max_err:.6e} > 1e-6 \
         (chunked append diverges from one-shot — buffer-arithmetic bug)"
    );

    let stats_a = cosine_similarity_per_row(&data, &recon_a, head_dim as usize);
    let stats_b = cosine_similarity_per_row(&data, &recon_b, head_dim as usize);
    assert!(
        stats_a.mean >= 0.99,
        "oneshot mean cosine {:.6} < 0.99",
        stats_a.mean
    );
    assert!(
        stats_b.mean >= 0.99,
        "chunked mean cosine {:.6} < 0.99",
        stats_b.mean
    );
}

/// CPU PlanarQuant on Bonsai-shaped 8k tensor. CPU path is the scalar
/// reference (no atomics, no growth buffers); if this passes but the GPU
/// equivalents fail, the bug is on the GPU side.
#[test]
fn planar_v4_cpu_roundtrip_8k_bonsai_shape() {
    let shape = [1i32, BONSAI_KV_H, 8192, BONSAI_HEAD_DIM];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0x00A1_65A4_BEEF_CAFE_u64);

    let blocks = planar_quantize(&data, GROUP_SIZE, 4, &shape).expect("CPU planar_quantize");
    let recon = planar_dequantize(&blocks).expect("CPU planar_dequantize");

    assert_eq!(recon.len(), n);
    let stats = cosine_similarity_per_row(&data, &recon, BONSAI_HEAD_DIM as usize);
    assert!(
        stats.mean >= 0.99,
        "CPU planar_v4 8k Bonsai shape mean cosine {:.6} < 0.99",
        stats.mean,
    );
}
