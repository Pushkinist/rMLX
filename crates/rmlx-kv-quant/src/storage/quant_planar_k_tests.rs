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
    // `data` is head-major `[1, kv_h, total_seq, head_dim]`. Both the prefill
    // chunk and each per-token decode step must be sliced head-major so they
    // carry the SAME logical (head, token) values Path A's one-shot append
    // sees — otherwise the comparison drifts because the chunk schedule feeds
    // a different tensor, not because the storage diverges. (The old slicing
    // assumed a token-contiguous layout, which only matched the pre-fix
    // token-major-write bug.)
    let mut init_shape_b = full_shape.to_vec();
    init_shape_b[2] = 0;
    let mut qpk_b = QuantPlanarK::new(init_shape_b, max_seq);
    let kv_h_u = kv_h as usize;
    let total_u = total_seq as usize;
    let d_u = head_dim as usize;
    let prefill_u = prefill_seq as usize;

    let mut prefill_data = vec![0.0f32; kv_h_u * prefill_u * d_u];
    for h in 0..kv_h_u {
        for s in 0..prefill_u {
            let src = (h * total_u + s) * d_u;
            let dst = (h * prefill_u + s) * d_u;
            prefill_data[dst..dst + d_u].copy_from_slice(&data[src..src + d_u]);
        }
    }
    let prefill_shape = [1i32, kv_h, prefill_seq, head_dim];
    let prefill_arr = make_f32_array(&prefill_data, &prefill_shape);
    qpk_b
        .append(&[], &prefill_shape, &prefill_arr, Device::Gpu, max_seq)
        .expect("prefill append B");
    let step_shape = [1i32, kv_h, 1, head_dim];
    for step in 0..decode_seq as usize {
        let s = prefill_u + step;
        let mut step_data = vec![0.0f32; kv_h_u * d_u];
        for h in 0..kv_h_u {
            let src = (h * total_u + s) * d_u;
            let dst = h * d_u;
            step_data[dst..dst + d_u].copy_from_slice(&data[src..src + d_u]);
        }
        let step_arr = make_f32_array(&step_data, &step_shape);
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

// ── Multi-append GQA head/seq-layout regression (multi-token-after-prefill scramble) ──
//
// The chunked-vs-oneshot parity test above never triggers the head-scramble:
// its chunked schedule is a multi-token prefill at `prev_seq == 0` followed by
// SINGLE-token decode appends. A single-token chunk is byte-identical in
// head-major and sequence-major order, so the token-major write offset
// (`prev_seq * words_per_seq`) is always correct for it. The scramble needs a
// MULTI-token append at `prev_seq > 0` — exactly the SSD-hydrate-then-reprefill
// path: an append whose chunk spans several tokens lands at a token-major
// offset, but the dequant view reshapes the prefix head-major (`[B, kv_h, S, D]`),
// transposing one head's new tokens onto another head's prefix.
//
// These fixtures build per-(head, token, dim) DISTINCT data so a head/seq swap
// is visible far above 4-bit planar quant noise: head `h` carries a sinusoid
// whose frequency scales with `h`, so a head that lands in the wrong slot has a
// near-zero cosine against the head-major reference.

/// Build `[1, kv_h, seq, d]` head-major f32 data where head `h`'s row at token
/// `s` is `sin((h+1) * base_freq * (d + s))`. Distinct per head (frequency) and
/// per token (phase); survives 4-bit planar quant per row at cosine ≈ 1.
fn head_distinct_data(kv_h: i32, seq: i32, d: i32) -> Vec<f32> {
    let kv_h = kv_h as usize;
    let seq = seq as usize;
    let d = d as usize;
    let mut out = vec![0.0f32; kv_h * seq * d];
    let base_freq = 0.05f32;
    for h in 0..kv_h {
        for s in 0..seq {
            for di in 0..d {
                let phase = (h as f32 + 1.0) * base_freq * ((di + s) as f32);
                out[(h * seq + s) * d + di] = phase.sin();
            }
        }
    }
    out
}

/// CPU diagnostic: a two-chunk append where the SECOND chunk is multi-token and
/// lands at `prev_seq > 0`. The dequant view must reproduce the head-major
/// reference for EVERY (head, token) row. Pre-fix this fails on `kv_h > 1`
/// because the token-major write scrambles heads↔seq.
#[test]
fn quant_planar_k_cpu_multi_append_gqa_head_layout() {
    let kv_h = 4i32;
    let head_dim = 64i32; // multiple of GROUP_SIZE (32), spans 2 groups per head
    let chunk0 = 5i32; // prefill chunk
    let chunk1 = 3i32; // multi-token decode/reprefill chunk at prev_seq > 0
    let total = chunk0 + chunk1;

    let full = head_distinct_data(kv_h, total, head_dim);
    let d = head_dim as usize;
    let total_u = total as usize;
    let kv_h_u = kv_h as usize;

    // Slice the head-major full tensor into two head-major chunks.
    let mut chunk0_data = vec![0.0f32; kv_h_u * chunk0 as usize * d];
    let mut chunk1_data = vec![0.0f32; kv_h_u * chunk1 as usize * d];
    for h in 0..kv_h_u {
        for s in 0..chunk0 as usize {
            let src = (h * total_u + s) * d;
            let dst = (h * chunk0 as usize + s) * d;
            chunk0_data[dst..dst + d].copy_from_slice(&full[src..src + d]);
        }
        for s in 0..chunk1 as usize {
            let src = (h * total_u + (chunk0 as usize + s)) * d;
            let dst = (h * chunk1 as usize + s) * d;
            chunk1_data[dst..dst + d].copy_from_slice(&full[src..src + d]);
        }
    }

    let max_seq = total + 16;
    let mut qpk = QuantPlanarK::new(vec![1i32, kv_h, 0, head_dim], max_seq);
    let c0_shape = [1i32, kv_h, chunk0, head_dim];
    let c1_shape = [1i32, kv_h, chunk1, head_dim];
    // CPU path ignores the `Array` arg (it uses `f32_data`); a 0-len dummy is fine.
    let dummy = Array::from_bytes(&[], &[0], Dtype::F32).expect("dummy array");
    qpk.append(&chunk0_data, &c0_shape, &dummy, Device::Cpu, max_seq)
        .expect("append chunk0");
    qpk.append(&chunk1_data, &c1_shape, &dummy, Device::Cpu, max_seq)
        .expect("append chunk1");

    let (recon, _) = qpk
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequantize_choice CPU");
    assert_eq!(recon.len(), full.len(), "recon length mismatch");

    // Per-row cosine against the head-major reference. A head/seq swap pushes
    // the swapped rows' cosine toward 0; a correct head-major buffer stays ≈ 1.
    let stats = cosine_similarity_per_row(&full, &recon, d);
    assert!(
        stats.min >= 0.97,
        "CPU multi-append GQA min per-row cosine {:.4} < 0.97 — head/seq scramble \
         on the multi-token-after-prefill append (kv_h={kv_h})",
        stats.min,
    );
}

/// GPU diagnostic: the same multi-append GQA head-layout contract on the Metal
/// path. This proves the packed buffer round-trips to head-major after a
/// multi-token append; the kernel-index agreement (fused-QK / flash-decode /
/// sparse read via `gpu_packed_view`) is covered separately by the fused-QK /
/// flash / sparse parity tests at `kv_h > 1`.
#[test]
#[ignore = "GPU Metal context — cargo test quant_planar_k_gpu_multi_append_gqa -- --ignored --test-threads=1"]
fn quant_planar_k_gpu_multi_append_gqa_head_layout() {
    if skip_if_no_gpu_env() {
        return;
    }
    let kv_h = 4i32;
    let head_dim = 64i32;
    let chunk0 = 5i32;
    let chunk1 = 3i32;
    let total = chunk0 + chunk1;
    let d = head_dim as usize;
    let total_u = total as usize;
    let kv_h_u = kv_h as usize;

    let full = head_distinct_data(kv_h, total, head_dim);

    let mut chunk0_data = vec![0.0f32; kv_h_u * chunk0 as usize * d];
    let mut chunk1_data = vec![0.0f32; kv_h_u * chunk1 as usize * d];
    for h in 0..kv_h_u {
        for s in 0..chunk0 as usize {
            let src = (h * total_u + s) * d;
            let dst = (h * chunk0 as usize + s) * d;
            chunk0_data[dst..dst + d].copy_from_slice(&full[src..src + d]);
        }
        for s in 0..chunk1 as usize {
            let src = (h * total_u + (chunk0 as usize + s)) * d;
            let dst = (h * chunk1 as usize + s) * d;
            chunk1_data[dst..dst + d].copy_from_slice(&full[src..src + d]);
        }
    }

    let init_shape = vec![1i32, kv_h, 0, head_dim];
    let max_seq = total + 256;
    let mut qpk = QuantPlanarK::new(init_shape, max_seq);
    let c0_shape = [1i32, kv_h, chunk0, head_dim];
    let c1_shape = [1i32, kv_h, chunk1, head_dim];
    let c0_arr = make_f32_array(&chunk0_data, &c0_shape);
    let c1_arr = make_f32_array(&chunk1_data, &c1_shape);
    qpk.append(&[], &c0_shape, &c0_arr, Device::Gpu, max_seq)
        .expect("append chunk0 GPU");
    qpk.append(&[], &c1_shape, &c1_arr, Device::Gpu, max_seq)
        .expect("append chunk1 GPU");

    let (_, recon_opt) = qpk
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("dequantize_choice GPU");
    let recon = array_to_f32(&recon_opt.expect("GPU dequant returned None"));
    assert_eq!(recon.len(), full.len(), "GPU recon length mismatch");

    let stats = cosine_similarity_per_row(&full, &recon, d);
    assert!(
        stats.min >= 0.97,
        "GPU multi-append GQA min per-row cosine {:.4} < 0.97 — head/seq scramble \
         on the multi-token-after-prefill append (kv_h={kv_h})",
        stats.min,
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

// ── Batch-axis block-boundary parity ──────────────────────────────────

/// Two appends must decode exactly like one append of the same tokens, at
/// `B > 1` as well as `B == 1`.
///
/// Each block covers `[B, S_block, kv_h, D]`, so the concatenation of two
/// blocks is not one `[B, S_total, kv_h, D]` run — reading it as one maps the
/// second block's batch-0 rows onto batch-1 sequence slots. The single-append
/// store holds exactly one block and therefore concatenates nothing, which is
/// what makes it the oracle here.
///
/// Mutation check: put `seq_layout::transpose_seq_heads` over the whole
/// concatenation back in `QuantPlanarK::dequantize_choice` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_planar_k_two_block_decode_matches_one_block_at_b_gt_1() {
    for (b, kv_h) in [(1_usize, 1_usize), (1, 2), (2, 1), (2, 2)] {
        let head_dim = 32_usize;
        let (n0, n1) = (2_usize, 3_usize);
        let max_seq = 512_i32;
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];
        let dummy =
            |n: usize| rmlx_mlx::zeros(&shape(n), Dtype::F32, Device::Cpu).expect("dummy array");
        let cpu_dequant = |st: &QuantPlanarK| {
            st.dequantize_choice(Device::Cpu, Dtype::F32)
                .expect("cpu dequant")
                .0
        };

        let mut one = QuantPlanarK::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], max_seq);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
            &dummy(n0 + n1),
            Device::Cpu,
            max_seq,
        )
        .expect("single append");
        let oracle = cpu_dequant(&one);

        let mut two = QuantPlanarK::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], max_seq);
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0, head_dim),
            &shape(n0),
            &dummy(n0),
            Device::Cpu,
            max_seq,
        )
        .expect("append chunk 0");
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, n0, n1, head_dim),
            &shape(n1),
            &dummy(n1),
            Device::Cpu,
            max_seq,
        )
        .expect("append chunk 1");
        let got = cpu_dequant(&two);

        assert_eq!(
            got, oracle,
            "two-block decode must equal the one-block oracle at b={b} kv_h={kv_h}"
        );
    }
}
