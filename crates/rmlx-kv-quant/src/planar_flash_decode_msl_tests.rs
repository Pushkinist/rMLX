//! Numerical parity tests for `planar_flash_decode`.
//!
//! Each test builds synthetic Q + K (encoded via the existing PlanarQuant
//! GPU packer) + V (plain f32) and compares the flash-decode output against a
//! fused-QK chain reference (`planar_fused_qk` + numpy-style softmax + plain
//! matmul SV).  Tolerance: max-abs ≤ 1e-4 on the f32 accumulator path.
//!
//! Tests are `#[ignore]`-gated so they only run inside a single-MLX GPU
//! context (see CLAUDE.md hard rule 8); enable via
//! `cargo test planar_flash_decode -- --include-ignored --test-threads=1`.

use super::*;
use crate::planar_fused_qk_msl::planar_fused_qk;
use crate::planarquant_msl::planar_quantize_v4_gpu;
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
#[allow(
    clippy::unwrap_used,
    reason = "test helper: chunks_exact(4) guarantees length"
)]
fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("array eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Reference SDPA chain (fused-QK reference chain): planar_fused_qk → softmax → matmul SV.
/// Computed in f32 host arithmetic over the dequantized K (already shipped
/// inside the fused-QK kernel) and the f32 V.
#[allow(
    clippy::needless_range_loop,
    reason = "test helper: indices cross two arrays (probs + scores / probs + v)"
)]
fn ref_flash_chain(
    scores: &[f32],
    v: &[f32],
    b: usize,
    n_q_heads: usize,
    kv_h: usize,
    kv_seq: usize,
    head_dim: usize,
) -> Vec<f32> {
    let heads_per_kv = n_q_heads / kv_h;
    let mut out = vec![0.0_f32; b * n_q_heads * head_dim];
    for bi in 0..b {
        for hq in 0..n_q_heads {
            let kv_h_idx = hq / heads_per_kv;
            let score_base = (bi * n_q_heads + hq) * kv_seq;
            // Softmax across kv_seq.
            let mut max_s = f32::NEG_INFINITY;
            for s in 0..kv_seq {
                let v = scores[score_base + s];
                if v > max_s {
                    max_s = v;
                }
            }
            let mut sum_exp = 0.0_f32;
            let mut probs = vec![0.0_f32; kv_seq];
            for s in 0..kv_seq {
                let e = (scores[score_base + s] - max_s).exp();
                probs[s] = e;
                sum_exp += e;
            }
            let inv = 1.0_f32 / sum_exp;
            for s in 0..kv_seq {
                probs[s] *= inv;
            }
            // SV: out[hq, d] = sum_s probs[s] * V[bi, kv_h_idx, s, d]
            let out_base = (bi * n_q_heads + hq) * head_dim;
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for s in 0..kv_seq {
                    let v_off = ((bi * kv_h + kv_h_idx) * kv_seq + s) * head_dim + d;
                    acc += probs[s] * v[v_off];
                }
                out[out_base + d] = acc;
            }
        }
    }
    out
}

fn run_parity(
    b: i32,
    kv_h: i32,
    heads_per_kv: i32,
    kv_seq: i32,
    head_dim: i32,
    k_seed: u64,
    q_seed: u64,
    v_seed: u64,
    tol: f32,
    tag: &str,
) {
    run_parity_with_v_dtype(
        b,
        kv_h,
        heads_per_kv,
        kv_seq,
        head_dim,
        k_seed,
        q_seed,
        v_seed,
        Dtype::F32,
        tol,
        tag,
    );
}

#[allow(
    clippy::too_many_arguments,
    reason = "test helper: parameter pack mirrors public dispatcher surface for parity coverage"
)]
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn run_parity_with_v_dtype(
    b: i32,
    kv_h: i32,
    heads_per_kv: i32,
    kv_seq: i32,
    head_dim: i32,
    k_seed: u64,
    q_seed: u64,
    v_seed: u64,
    v_dtype: Dtype,
    tol: f32,
    tag: &str,
) {
    let n_q_heads = kv_h * heads_per_kv;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, k_seed);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, q_seed);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let v_shape = [b, kv_h, kv_seq, head_dim];
    let v_n: usize = v_shape.iter().map(|&d| d as usize).product();
    let v_data_f32 = lcg_data(v_n, v_seed);
    let v_arr_f32 = make_f32_array(&v_data_f32, &v_shape);

    // Build the V array in the requested dtype.  For non-f32, round-trip
    // through MLX `astype` and pull bytes back so the reference sees the
    // same quantised values the kernel will read.
    let (v_arr, v_data_for_ref) = match v_dtype {
        Dtype::F32 => (v_arr_f32, v_data_f32),
        Dtype::Bf16 | Dtype::F16 => {
            let v_cast = v_arr_f32
                .astype(v_dtype, Device::Gpu)
                .expect("V astype to native dtype");
            // Round-trip back to f32 for the reference so the parity check
            // accounts for the dtype's representable-value rounding.
            let v_f32_round = v_cast
                .astype(Dtype::F32, Device::Gpu)
                .expect("V astype back to f32 for reference");
            let round = array_to_f32(&v_f32_round);
            (v_cast, round)
        }
        Dtype::U8 | Dtype::U32 | Dtype::I32 => {
            panic!("unsupported V dtype for parity test: {v_dtype:?}")
        }
    };

    // K packing is sequence-major (`[B, S, kv_h, D]`) — the layout the
    // fused-QK / flash-decode kernels index. Transpose the head-major `k_arr`
    // heads↔seq and materialize before packing so the packed buffer matches.
    let k_seq = k_arr
        .transpose(&[0, 2, 1, 3], Device::Gpu)
        .expect("transpose k seq-major")
        .contiguous(Device::Gpu)
        .expect("contiguous k seq-major");
    let (codes, scales, rot32) =
        planar_quantize_v4_gpu(&k_seq, Device::Gpu).expect("planar_quantize_v4_gpu");

    // ── Flash output ─────────────────────────────────────────────────────
    let flash = planar_flash_decode_sdpa(
        &q_arr,
        &codes,
        &scales,
        &rot32,
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
    let flash_vec = array_to_f32(&flash);

    // ── Reference: fused-QK → host softmax → host SV ─────────────────────
    let scores = planar_fused_qk(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        Device::Gpu,
    )
    .expect("planar_fused_qk");
    let scores_vec = array_to_f32(&scores);

    let ref_out = ref_flash_chain(
        &scores_vec,
        &v_data_for_ref,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
    );

    assert_eq!(flash_vec.len(), ref_out.len());
    let max_err = flash_vec
        .iter()
        .zip(ref_out.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "[planar_flash_decode parity] {tag} v_dtype={v_dtype:?} max_abs_err={max_err:.6e} tol={tol:.6e}"
    );
    assert!(
        max_err < tol,
        "{tag} (v_dtype={v_dtype:?}): planar_flash_decode max abs error {max_err:.6e} > tol {tol:.6e}"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_flash_decode -- --include-ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_flash_decode_v4_matches_t25_chain_gqa_128() {
    if skip_if_no_gpu_env() {
        return;
    }
    // GQA: 2 KV heads × 4 query heads each = 8 q-heads. kv_seq=64 = 1 tile.
    run_parity(
        1,
        2,
        4,
        64,
        128,
        0xCAFE_1234,
        0xBEEF_5678,
        0x1357_2468,
        1e-4,
        "v4 GQA head_dim=128 single-tile",
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_flash_decode -- --include-ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_flash_decode_v4_matches_t25_chain_multi_tile() {
    if skip_if_no_gpu_env() {
        return;
    }
    // kv_seq=192 = 3 tiles of TILE_SIZE=64.  This exercises the LSE merge in
    // pass 2 — the only test path that touches both partial slots and the
    // cross-tile correction.
    run_parity(
        1,
        2,
        4,
        192,
        128,
        0xAAAA_BBBB,
        0xCCCC_DDDD,
        0xEEEE_FFFF,
        1e-4,
        "v4 GQA head_dim=128 multi-tile (3)",
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_flash_decode -- --include-ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_flash_decode_v4_matches_t25_chain_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Gemma4-26B class head_dim=256.
    run_parity(
        1,
        2,
        2,
        128,
        256,
        0x2560_0001,
        0x2560_0002,
        0x2560_0003,
        1e-4,
        "v4 GQA head_dim=256 multi-tile (2)",
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_flash_decode -- --include-ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_flash_decode_v4_matches_t25_chain_bf16_v() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Proves the HIGH-finding fix: V passed in native bf16 (no f32 astype
    // upcast in the dispatcher).  The reference V is bf16-round-tripped so
    // the comparison sees the same quantised values the kernel reads; the
    // resulting max-abs error stays well under the spec 1e-4 bound.
    run_parity_with_v_dtype(
        1,
        2,
        4,
        192,
        128,
        0xBF16_0001,
        0xBF16_0002,
        0xBF16_0003,
        Dtype::Bf16,
        1e-4,
        "v4 GQA head_dim=128 multi-tile (3) bf16 V",
    );
}

#[test]
fn planar_flash_decode_rejects_non_pow2_head_dim() {
    // head_dim=96 satisfies %32 but not pow-2 — exercises the pow-2 guard.
    let dummy_q = make_f32_array(&[0.0_f32], &[1]);
    let dummy_codes = Array::from_bytes(&[0u8, 0, 0, 0], &[1], Dtype::U32).expect("codes");
    let dummy_scales = make_f32_array(&[0.0_f32], &[1]);
    let dummy_rot = Array::from_bytes(&[0u8, 0, 0, 0], &[1], Dtype::U32).expect("rot");
    let dummy_v = make_f32_array(&[0.0_f32], &[1]);

    let err = planar_flash_decode_sdpa(
        &dummy_q,
        &dummy_codes,
        &dummy_scales,
        &dummy_rot,
        &dummy_v,
        None,
        1,
        1,
        1,
        96,
        1,
        4,
        1.0,
        Device::Cpu,
    )
    .expect_err("planar_flash_decode_sdpa should reject non-pow-2 head_dim");
    let msg = err.to_string();
    assert!(
        msg.contains("power of two"),
        "expected pow-2 guard error, got: {msg}"
    );
}

#[test]
fn a_planar_flash_decode_dispatch_count_initially_zero() {
    // Counter must be zero before any GPU dispatch in a fresh test binary.
    // The "then increases" half of the previous combined name is covered by
    // `planar_flash_decode_dispatch_count_increments_on_gpu` below (GPU-only,
    // `#[ignore]`-gated alongside the other Metal parity tests).
    //
    // The `a_` prefix forces this test to sort lexically first so it runs
    // before any GPU-tagged sibling under `--test-threads=1 --include-ignored`
    // — cargo test runs tests in name-sort order within a binary, and the
    // dispatch counter is a process-global atomic that any earlier GPU test
    // would already have incremented.
    assert_eq!(
        planar_flash_decode_dispatch_count(),
        0,
        "counter must be zero before any GPU dispatch in a fresh test binary"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_flash_decode -- --include-ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_flash_decode_dispatch_count_increments_on_gpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Reuse the smallest GQA parity config so the increment proof is cheap.
    // After at least one P1 dispatch the counter must be strictly positive.
    let before = planar_flash_decode_dispatch_count();
    run_parity(
        1,
        2,
        4,
        64,
        128,
        0x1111_2222,
        0x3333_4444,
        0x5555_6666,
        1e-4,
        "dispatch counter increment",
    );
    let after = planar_flash_decode_dispatch_count();
    assert!(
        after > before,
        "dispatch counter did not advance: before={before}, after={after}"
    );
}

#[test]
fn planar_flash_decode_enabled_is_callable() {
    // Smoke: the gate accessor must be callable from any thread without
    // panicking.  Value is OnceLock-cached after first read; this is the
    // first read in non-ignored test order, so it latches the env state at
    // test-run time.
    let _: bool = planar_flash_decode_enabled();
}

// ── Bit-exactness vs the production split chain ───────────────────────────
//
// `planar_k_flash_over_store` has exactly two arms over one packed store:
// the single-pass flash kernel, and the split chain
// (`planar_fused_qk` → mask add → `softmax_precise` → GQA matmul). The
// helper below runs *both* arms on the same inputs and reports how many
// output elements differ bit-for-bit, so the two arms are never assumed
// equal from a run in which only one of them executed.

/// Run both `planar_k_flash_over_store` arms over one packed K and return
/// `(n_elements, n_bitwise_differences, max_abs_err)`.
///
/// Mirrors the production chain step for step, including the f32 softmax and
/// the GQA reshape/matmul, so the numbers describe the arms operators
/// actually select between — not a host-side approximation of them.
#[allow(
    clippy::too_many_arguments,
    reason = "test helper: parameter pack mirrors the dispatcher's shape surface"
)]
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn flash_vs_split_chain(
    b: i32,
    kv_h: i32,
    heads_per_kv: i32,
    kv_seq: i32,
    head_dim: i32,
    seed: u64,
) -> (usize, usize, f32) {
    use rmlx_mlx::{matmul, softmax_precise};

    let n_q_heads = kv_h * heads_per_kv;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let device = Device::Gpu;

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_arr = make_f32_array(&lcg_data(k_n, seed), &k_shape);
    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_arr = make_f32_array(&lcg_data(q_n, seed ^ 0x1111_1111), &q_shape);
    let v_shape = [b, kv_h, kv_seq, head_dim];
    let v_n: usize = v_shape.iter().map(|&d| d as usize).product();
    let v_arr = make_f32_array(&lcg_data(v_n, seed ^ 0x2222_2222), &v_shape)
        .astype(Dtype::Bf16, device)
        .expect("V bf16");

    // Pack K seq-major, the layout both kernels index.
    let k_seq = k_arr
        .transpose(&[0, 2, 1, 3], device)
        .expect("transpose k seq-major")
        .contiguous(device)
        .expect("contiguous k seq-major");
    let (codes, scales, rot32) =
        planar_quantize_v4_gpu(&k_seq, device).expect("planar_quantize_v4_gpu");

    // Arm A — single-pass flash kernel.
    let flash = planar_flash_decode_sdpa(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        &v_arr,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        device,
    )
    .expect("planar_flash_decode_sdpa");

    // Arm B — split chain, verbatim from `planar_k_flash_over_store`.
    let scores = planar_fused_qk(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        device,
    )
    .expect("planar_fused_qk");
    let probs = softmax_precise(&scores, -1, device).expect("softmax_precise");
    let probs_g = probs
        .reshape(&[b, kv_h, heads_per_kv, 1, kv_seq], device)
        .expect("probs reshape");
    let v_g = v_arr
        .reshape(&[b, kv_h, 1, kv_seq, head_dim], device)
        .expect("v reshape");
    let out_g = matmul(&probs_g, &v_g, device).expect("GQA matmul");
    let split = out_g
        .reshape(&[b, n_q_heads, 1, head_dim], device)
        .expect("out reshape");

    let a = array_to_f32(&flash.astype(Dtype::F32, device).expect("flash f32"));
    let s = array_to_f32(&split.astype(Dtype::F32, device).expect("split f32"));
    assert_eq!(a.len(), s.len(), "arm output lengths differ");
    let mut diffs = 0_usize;
    let mut max_err = 0.0_f32;
    for (x, y) in a.iter().zip(s.iter()) {
        if x.to_bits() != y.to_bits() {
            diffs += 1;
        }
        let e = (x - y).abs();
        if e > max_err {
            max_err = e;
        }
    }
    (a.len(), diffs, max_err)
}

/// The flash kernel and the split chain are **not** bit-exact.
///
/// Both arms decode the same packed K, but the flash kernel folds the
/// softmax into a per-tile online (log-sum-exp) reduction while the split
/// chain materialises the full score row and calls `softmax_precise`. The
/// two summation orders differ, so the results differ in the low mantissa
/// bits. This test pins that as a measured fact across short and long
/// `kv_seq` and both test-target head shapes, because the surrounding docs
/// used to assert byte-identity from a single cell.
///
/// Failing *upward* (zero differences everywhere) is also a real signal —
/// it would mean one arm stopped running, which is exactly the inert A/B
/// this file exists to prevent.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_flash_decode -- --include-ignored --test-threads=1"]
fn planar_flash_decode_is_not_bit_exact_vs_split_chain() {
    if skip_if_no_gpu_env() {
        return;
    }
    // (kv_h, heads_per_kv, head_dim): Bonsai-8B-shaped GQA at head_dim=128
    // and a kv_h=1 single-KV-head shape at head_dim=256.
    let shapes = [(8_i32, 4_i32, 128_i32), (1_i32, 8_i32, 256_i32)];
    let contexts = [64_i32, 512, 4096];
    let mut any_diff = false;
    for (kv_h, heads_per_kv, head_dim) in shapes {
        for kv_seq in contexts {
            let seed = 0x9E37_79B9_u64
                .wrapping_mul(kv_seq as u64)
                .wrapping_add(head_dim as u64);
            let (n, diffs, max_err) =
                flash_vs_split_chain(1, kv_h, heads_per_kv, kv_seq, head_dim, seed);
            eprintln!(
                "[planar flash vs split] kv_h={kv_h} heads_per_kv={heads_per_kv} \
                 head_dim={head_dim} kv_seq={kv_seq}: {diffs}/{n} elements differ \
                 bitwise, max_abs_err={max_err:.3e}"
            );
            assert!(
                max_err < 1e-3,
                "kv_h={kv_h} head_dim={head_dim} kv_seq={kv_seq}: arms disagree by \
                 {max_err:.3e} — that is a correctness bug, not rounding"
            );
            if diffs > 0 {
                any_diff = true;
            }
        }
    }
    assert!(
        any_diff,
        "every cell matched bit-for-bit — either the kernel became bit-exact \
         (update the docs) or one arm did not run (the A/B is inert)"
    );
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
fn hdr_probe_snapshot_matches_builder() {
    assert_eq!(
        p1_header_v4().expect("planar flash P1 header"),
        include_str!("metal/probes/planar_flash_decode_p1.hdr.metal"),
        "stale snapshot: refresh metal/probes/planar_flash_decode_p1.hdr.metal"
    );
}
