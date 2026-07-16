//! Sparse-attention parity + dispatch-counter tests.
//!
//! # Parity contract
//!
//! Synthetic PlanarQuant K + V + Q.  Compare:
//!
//! * Dense reference: `planar_flash_decode_sdpa` (planar-flash chain).
//! * Sparse: Phase 1 (top-K-per-tile scoring) → CPU/host head-threshold
//!   (the K-th largest score across all tile-tops per `(B, H)`) →
//!   Phase 2 (re-decode K/V on survivors + online softmax) → P2 LSE merge.
//!
//! At 95% mass budget, per-row cosine ≥ 0.99 is the gate.
//!
//! # Configs
//!
//! * single-tile: kv_seq=64, head_dim=128
//! * multi-tile (3 tiles × 64 = 192 kv_seq), head_dim=128
//! * multi-tile head_dim=256
//!
//! GPU tests are `#[ignore]`-gated; run with
//! `cargo test -p rmlx-kv-quant sparse_attn -- --include-ignored --test-threads=1`.

use super::phase1_score_msl::{phase1_score, phase1_score_dispatch_count, TOP_PER_TILE};
use super::phase2_sparse_attend_msl::{
    phase2_lse_merge, phase2_sparse_attend, phase2_sparse_attend_dispatch_count,
};
use crate::planar_flash_decode_msl::planar_flash_decode_sdpa;
use crate::planarquant_msl::planar_quantize_v4_gpu;
use crate::test_utils::{cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(clippy::expect_used, clippy::unwrap_used, reason = "test helper")]
fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("array eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Compute the per-`(B, H)` head_threshold from Phase-1's `tile_top_scores`.
///
/// `tile_top_scores` is `[n_tiles, n_bh, TOP_PER_TILE]` f32 (descending
/// per tile).  Strategy: flatten across tiles per `(B, H)`, sort
/// descending, take the K-th largest score.  When fewer than K
/// candidates exist (small kv_seq), fall back to the smallest available
/// score so every token survives.
///
/// This mirrors the "CPU bridge" the multi-turboquant reference uses
/// between PHASE1_SCORE_KERNEL and PHASE2_SPARSE_ATTEND_KERNEL.
fn cpu_head_threshold(tile_top_scores: &[f32], n_tiles: usize, n_bh: usize, k: usize) -> Vec<f32> {
    let mut thresholds = vec![f32::NEG_INFINITY; n_bh];
    let top_per_tile = TOP_PER_TILE as usize;
    for bh in 0..n_bh {
        let mut all: Vec<f32> = Vec::with_capacity(n_tiles * top_per_tile);
        for t in 0..n_tiles {
            for k_slot in 0..top_per_tile {
                let v = tile_top_scores[(t * n_bh + bh) * top_per_tile + k_slot];
                if v.is_finite() {
                    all.push(v);
                }
            }
        }
        all.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let take = k.min(all.len()).max(1) - 1;
        thresholds[bh] = if all.is_empty() {
            f32::NEG_INFINITY
        } else {
            all[take]
        };
    }
    thresholds
}

/// Choose `k` so that the survivors cover ~`mass_frac` of the softmax mass.
///
/// Heuristic: pick `k` = `ceil(mass_frac * kv_seq)`.  At 95% mass + the
/// typical exponential decay of attention scores, this consistently
/// captures the right tail.  Empirically validated against the dense
/// reference (cosine ≥ 0.99 in all three synthetic configs).
fn budget_for_mass(kv_seq: usize, mass_frac: f32) -> usize {
    ((kv_seq as f32) * mass_frac).ceil() as usize
}

#[allow(
    clippy::too_many_arguments,
    clippy::expect_used,
    reason = "test harness: parameter pack mirrors dispatcher signatures"
)]
fn run_parity(
    b: i32,
    kv_h: i32,
    heads_per_kv: i32,
    kv_seq: i32,
    head_dim: i32,
    k_seed: u64,
    q_seed: u64,
    v_seed: u64,
    mass_frac: f32,
    cos_tol: f32,
    tag: &str,
) {
    let n_q_heads = kv_h * heads_per_kv;
    let n_bh = b * n_q_heads;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    // Sparse-friendly Q/K construction:
    //   K is LCG noise scaled to ~Gaussian(0, 1).
    //   Q is LCG noise scaled the same way, BUT with one designated K row
    //   per (b, kv_h) "planted" to align with Q's average across the GQA
    //   group — this makes its QK score dominate (Q · K_planted ≈ ||Q||²),
    //   while other K rows have score ~ O(sqrt(head_dim)) by Gaussian dot
    //   product.  Softmax mass concentrates on the planted row → top-4-per-
    //   tile candidate set always includes it → sparse output matches dense
    //   to within softmax-tail rounding.
    //
    // mass_frac controls the budget the CPU bridge will pick from
    // tile_top_scores; the planted row's score is always #1 globally so any
    // budget ≥ 1 keeps it.

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, q_seed);
    let q_arr = make_f32_array(&q_data, &q_shape);

    // Average Q across the GQA group per (b, kv_h) to seed the planted K row.
    let mut k_data = lcg_data((b * kv_h * kv_seq * head_dim) as usize, k_seed);
    let heads_per_kv_u = heads_per_kv as usize;
    let head_dim_u = head_dim as usize;
    let kv_seq_u = kv_seq as usize;
    let kv_h_u = kv_h as usize;
    let b_u = b as usize;
    // Planted K row index per (b, kv_h) — fix at row 0 of the FIRST tile so
    // the smoke test exercises Phase 1's top-1 path; second batch / kv_h pair
    // uses row (kv_seq / 2) so the multi-tile config exercises a non-zero tile.
    let planted_row: usize = (kv_seq as usize / 2).min(kv_seq_u - 1);
    let q_amplify: f32 = 8.0;
    for bi in 0..b_u {
        for h in 0..kv_h_u {
            let q_h_base_start = h * heads_per_kv_u;
            let mut avg = vec![0.0f32; head_dim_u];
            for hq_off in 0..heads_per_kv_u {
                let hq = q_h_base_start + hq_off;
                let q_off = ((bi * n_q_heads as usize) + hq) * head_dim_u;
                for d in 0..head_dim_u {
                    avg[d] += q_data[q_off + d];
                }
            }
            let inv = 1.0f32 / (heads_per_kv as f32);
            let k_off = ((bi * kv_h_u + h) * kv_seq_u + planted_row) * head_dim_u;
            for d in 0..head_dim_u {
                k_data[k_off + d] = avg[d] * inv * q_amplify;
            }
        }
    }
    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_arr = make_f32_array(&k_data, &k_shape);

    let v_shape = [b, kv_h, kv_seq, head_dim];
    let v_n: usize = v_shape.iter().map(|&d| d as usize).product();
    let v_data = lcg_data(v_n, v_seed);
    let v_arr = make_f32_array(&v_data, &v_shape);

    // ── PlanarQuant pack K (sequence-major) ──────────────────────────────
    // The fused-QK / flash-decode / sparse phase-1/2 kernels index K
    // sequence-major (`[B, S, kv_h, D]`); transpose the head-major `k_arr`
    // heads↔seq and materialize before packing so the packed buffer matches.
    let k_seq = k_arr
        .transpose(&[0, 2, 1, 3], Device::Gpu)
        .expect("transpose k seq-major")
        .contiguous(Device::Gpu)
        .expect("contiguous k seq-major");
    let (k_codes, k_scales, k_rot32) =
        planar_quantize_v4_gpu(&k_seq, Device::Gpu).expect("planar_quantize_v4_gpu");

    // ── Dense reference (planar flash decode) ────────────────────────────
    let dense = planar_flash_decode_sdpa(
        &q_arr,
        &k_codes,
        &k_scales,
        &k_rot32,
        &v_arr,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        Device::Gpu,
    )
    .expect("planar_flash_decode_sdpa");
    let dense_vec = array_to_f32(&dense);

    // ── Phase 1 ──────────────────────────────────────────────────────────
    let p1 = phase1_score(
        &q_arr,
        &k_codes,
        &k_scales,
        &k_rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("phase1_score");

    // ── CPU head_threshold ──────────────────────────────────────────────
    let tts_vec = array_to_f32(&p1.tile_top_scores);
    let budget = budget_for_mass(kv_seq as usize, mass_frac);
    let thr_vec = cpu_head_threshold(&tts_vec, p1.n_tiles as usize, n_bh as usize, budget);
    let head_threshold = make_f32_array(&thr_vec, &[n_bh]);

    // ── Phase 2 ──────────────────────────────────────────────────────────
    let p2 = phase2_sparse_attend(
        &q_arr,
        &k_codes,
        &k_scales,
        &k_rot32,
        &v_arr,
        &p1.all_scores,
        &head_threshold,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        p1.n_tiles,
        scale,
        Device::Gpu,
    )
    .expect("phase2_sparse_attend");

    let merged = phase2_lse_merge(
        &p2.partial_o,
        &p2.tile_lse,
        b,
        n_q_heads,
        head_dim,
        p1.n_tiles,
        Device::Gpu,
    )
    .expect("phase2_lse_merge");
    let sparse_vec = array_to_f32(&merged);

    assert_eq!(sparse_vec.len(), dense_vec.len());
    let cos = cosine_similarity_per_row(&dense_vec, &sparse_vec, head_dim as usize);
    eprintln!(
        "[sparse_attn parity] {tag} kv_seq={kv_seq} head_dim={head_dim} \
         budget={budget} mass={mass_frac:.2} cos_mean={:.4} cos_min={:.4} n_rows={}",
        cos.mean, cos.min, cos.n_rows
    );

    assert!(
        cos.min >= cos_tol,
        "{tag}: sparse-vs-dense per-row cosine min {:.4} < tol {:.4} (mean {:.4})",
        cos.min,
        cos_tol,
        cos.mean
    );
}

// ── GPU parity tests ─────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test sparse_attn -- --include-ignored --test-threads=1"]
fn sparse_attn_parity_single_tile_head_dim_128() {
    if skip_if_no_gpu_env() {
        return;
    }
    // 1 tile = 64 tokens, head_dim=128.  GQA: 2 kv heads × 4 q-per-kv = 8 q-heads.
    run_parity(
        1,
        2,
        4,
        64,
        128,
        0xCAFE_0001,
        0xBEEF_0001,
        0x1357_0001,
        0.95,
        0.99,
        "single-tile dim128",
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test sparse_attn -- --include-ignored --test-threads=1"]
fn sparse_attn_parity_multi_tile_head_dim_128() {
    if skip_if_no_gpu_env() {
        return;
    }
    // 3 tiles × 64 = 192 tokens (exercises P2 LSE merge across tiles).
    run_parity(
        1,
        2,
        4,
        192,
        128,
        0xAAAA_0002,
        0xCCCC_0002,
        0xEEEE_0002,
        0.95,
        0.99,
        "multi-tile (3) dim128",
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test sparse_attn -- --include-ignored --test-threads=1"]
fn sparse_attn_parity_multi_tile_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    // 3 tiles × 64 = 192 tokens, Gemma4-26B-class head_dim=256.
    run_parity(
        1,
        2,
        2,
        192,
        256,
        0xF00D_0003,
        0xDEAD_0003,
        0xBEEF_0003,
        0.95,
        0.99,
        "multi-tile (3) dim256",
    );
}

// ── Dispatch counter tests (GPU) ─────────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test sparse_attn -- --include-ignored --test-threads=1"]
fn sparse_attn_dispatch_counters_increment() {
    if skip_if_no_gpu_env() {
        return;
    }
    let p1_before = phase1_score_dispatch_count();
    let p2_before = phase2_sparse_attend_dispatch_count();
    // Run a single synthetic call — same shape as the first parity test.
    run_parity(
        1,
        2,
        4,
        64,
        128,
        0xC0DE_0004,
        0xCAFE_0004,
        0xFACE_0004,
        0.95,
        0.99,
        "dispatch-counter probe",
    );
    let p1_after = phase1_score_dispatch_count();
    let p2_after = phase2_sparse_attend_dispatch_count();
    assert!(
        p1_after > p1_before,
        "PHASE1_DISPATCHES did not increment ({p1_before} -> {p1_after})"
    );
    assert!(
        p2_after > p2_before,
        "PHASE2_DISPATCHES did not increment ({p2_before} -> {p2_after})"
    );
}

// ── Phase-1 structural smoke (no GPU; build-only assertion via compile) ──────
//
// These compile-time / cheap CPU-side sanity checks do not dispatch a Metal
// kernel; they assert the helpers that bridge P1 → P2 behave per spec.

#[test]
fn cpu_head_threshold_picks_kth_largest() {
    // n_tiles=2, n_bh=1, TOP_PER_TILE=4: 8 scores [10,9,8,7] and [6,5,4,3].
    // k=3 → 3rd largest = 8.
    let scores: Vec<f32> = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0];
    let thr = cpu_head_threshold(&scores, 2, 1, 3);
    assert_eq!(thr.len(), 1);
    assert!(
        (thr[0] - 8.0).abs() < 1e-6,
        "expected threshold 8.0, got {}",
        thr[0]
    );
}

#[test]
fn cpu_head_threshold_falls_back_when_few_survivors() {
    // 2 finite scores + 6 -inf padding; k=4 must clamp to smallest finite.
    let mut scores: Vec<f32> = vec![10.0, 9.0];
    scores.extend(std::iter::repeat_n(f32::NEG_INFINITY, 6));
    let thr = cpu_head_threshold(&scores, 2, 1, 4);
    assert!(
        (thr[0] - 9.0).abs() < 1e-6,
        "expected fallback to smallest finite (9.0), got {}",
        thr[0]
    );
}

#[test]
fn budget_for_mass_rounds_up() {
    assert_eq!(budget_for_mass(100, 0.95), 95);
    assert_eq!(budget_for_mass(64, 0.95), 61); // ceil(60.8) = 61
    assert_eq!(budget_for_mass(1, 0.95), 1);
}

/// Probe header snapshots must equal what the builders emit.
///
/// `make check-metal-compiles` prepends these snapshots to the kernel bodies.
/// A builder that changes a constant's value, or drops one, without the
/// snapshot being refreshed leaves the probe compiling text production no
/// longer emits — the gate would keep passing while checking the wrong thing.
/// Equality here turns that drift into a hard failure.
#[allow(
    clippy::expect_used,
    reason = "a header that fails to build is itself the drift this test guards"
)]
#[test]
fn hdr_probe_snapshots_match_builders() {
    assert_eq!(
        crate::sparse_attn::phase1_score_msl::p1_header().expect("phase1 header"),
        include_str!("../metal/probes/sparse_attn_phase1_score.hdr.metal"),
        "stale snapshot: refresh ../metal/probes/sparse_attn_phase1_score.hdr.metal"
    );
    assert_eq!(
        crate::sparse_attn::phase2_sparse_attend_msl::p2_header().expect("phase2 header"),
        include_str!("../metal/probes/sparse_attn_phase2_attend.hdr.metal"),
        "stale snapshot: refresh ../metal/probes/sparse_attn_phase2_attend.hdr.metal"
    );
}
