//! GPU parity tests for the rotor flash-decode kernel (BITS in {3, 4}).
//!
//! The oracle is the codec's own CPU dequant
//! ([`crate::rotorquant::rotor3_decode`] / [`rotor4_decode`]) fed through a
//! scalar reference attention chain. That is exactly what the production path
//! did before this kernel existed, so a match proves the kernel reproduces the
//! path it replaces — not merely that it is self-consistent.

use super::*;
use crate::clifford::make_rotor_table;
use crate::rotorquant::{rotor3_decode, rotor3_encode, rotor4_decode, rotor4_encode};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

// ── Test helpers ──────────────────────────────────────────────────────────────

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_u32_array(data: &[u32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::U32).expect("make_u32_array")
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

/// Scalar reference attention over an already-dequantized K.
///
/// `k_deq` is sequence-major (`[(s * kv_h + h), head_dim]`) — the layout the
/// rotor store accumulates and the kernel reads. `v` is head-major
/// (`[b, kv_h, kv_seq, head_dim]`) — the bf16 mirror's layout.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "test reference: all indices derived from the shape params under test"
)]
fn ref_attention(
    q: &[f32],
    k_deq: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    n_q_heads: usize,
    kv_h: usize,
    kv_seq: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let heads_per_kv = n_q_heads / kv_h;
    let mut out = vec![0.0_f32; n_q_heads * head_dim];
    for hq in 0..n_q_heads {
        let h = hq / heads_per_kv;

        // Scores over the whole prefix.
        let mut scores = vec![0.0_f32; kv_seq];
        for (s, score) in scores.iter_mut().enumerate() {
            let k_row = (s * kv_h + h) * head_dim;
            let mut acc = 0.0_f32;
            for d in 0..head_dim {
                acc += q[hq * head_dim + d] * k_deq[k_row + d];
            }
            *score = acc * scale;
        }
        if let Some(m) = mask {
            // Additive mask, laid out [b, n_q_heads, 1, kv_seq] with b == 1.
            for (s, score) in scores.iter_mut().enumerate() {
                *score += m[hq * kv_seq + s];
            }
        }

        // Softmax (max-subtracted, matching the kernel's online form).
        let mut m = f32::NEG_INFINITY;
        for &s in &scores {
            if s > m {
                m = s;
            }
        }
        let mut denom = 0.0_f32;
        for s in &mut scores {
            *s = (*s - m).exp();
            denom += *s;
        }

        // Weighted V sum.
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for (s, &p) in scores.iter().enumerate() {
                acc += p * v[(h * kv_seq + s) * head_dim + d];
            }
            out[hq * head_dim + d] = acc / denom;
        }
    }
    out
}

/// Encode `k_seq_major` with the rotor codec on CPU and return the GPU-resident
/// packed buffers plus the CPU dequant oracle.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor_encode_for_test(
    k_seq_major: &[f32],
    head_dim: usize,
    bits: u8,
) -> (Array, Array, Array, Array, Vec<f32>) {
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(0, 0, n_groups);

    let (codes, scales, norms) = if bits == 3 {
        rotor3_encode(k_seq_major, &rotors, head_dim).expect("rotor3_encode")
    } else {
        rotor4_encode(k_seq_major, &rotors, head_dim).expect("rotor4_encode")
    };
    let dequant = if bits == 3 {
        rotor3_decode(&codes, &scales, &norms, &rotors, head_dim).expect("rotor3_decode")
    } else {
        rotor4_decode(&codes, &scales, &norms, &rotors, head_dim).expect("rotor4_decode")
    };

    (
        make_u32_array(&codes, &[codes.len() as i32]),
        make_f32_array(&scales, &[scales.len() as i32]),
        make_f32_array(&norms, &[norms.len() as i32]),
        make_f32_array(&rotors, &[rotors.len() as i32]),
        dequant,
    )
}

/// Run the kernel against the CPU-dequant oracle for one shape, with no mask.
fn run_oracle(bits: u8, kv_h: usize, heads_per_kv: usize, kv_seq: usize, head_dim: usize) -> bool {
    run_oracle_masked(bits, kv_h, heads_per_kv, kv_seq, head_dim, false)
}

/// Run the kernel against the CPU-dequant oracle for one shape.
///
/// When `with_mask`, feeds a per-(q_head, key) additive mask with a distinct
/// value in every cell, so a transposed or mis-strided mask index in either the
/// kernel or the dispatcher changes the result instead of cancelling out.
///
/// Returns `false` when the GPU is unavailable so the caller can skip.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
#[allow(clippy::too_many_arguments)]
fn run_oracle_masked(
    bits: u8,
    kv_h: usize,
    heads_per_kv: usize,
    kv_seq: usize,
    head_dim: usize,
    with_mask: bool,
) -> bool {
    if skip_if_no_gpu_env() {
        return false;
    }
    let b = 1_usize;
    let n_q_heads = kv_h * heads_per_kv;
    let n_tokens = kv_seq * kv_h;

    let q = lcg_data(n_q_heads * head_dim, 0xA11CE);
    // K is generated in the store's sequence-major order: token (s, h) at row
    // (s * kv_h + h).
    let k = lcg_data(n_tokens * head_dim, 0xB0B);
    let v = lcg_data(n_tokens * head_dim, 0xC0FFEE);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let (codes, scales, norms, rotors, k_deq) = rotor_encode_for_test(&k, head_dim, bits);
    let q_arr = make_f32_array(&q, &[b as i32, n_q_heads as i32, 1, head_dim as i32]);
    let v_arr = make_f32_array(&v, &[b as i32, kv_h as i32, kv_seq as i32, head_dim as i32]);

    // Asymmetric in both axes and never zero, so a (hq, s) index swap or a wrong
    // stride shifts the scores rather than landing on an equal value. Includes a
    // -inf-ish cell to exercise full suppression through the online softmax.
    let mask_vec: Option<Vec<f32>> = with_mask.then(|| {
        (0..n_q_heads * kv_seq)
            .map(|i| {
                let hq = i / kv_seq;
                let s = i % kv_seq;
                if hq == 0 && s == kv_seq / 2 {
                    -1.0e30
                } else {
                    (hq as f32) * 0.25 - (s as f32) * 0.03125
                }
            })
            .collect()
    });
    let mask_arr = mask_vec
        .as_ref()
        .map(|m| make_f32_array(m, &[b as i32, n_q_heads as i32, 1, kv_seq as i32]));

    let out = match bits {
        3 => rotor_flash_decode_sdpa::<3>(
            &q_arr,
            &codes,
            &scales,
            &norms,
            &rotors,
            &v_arr,
            mask_arr.as_ref(),
            b as i32,
            kv_h as i32,
            kv_seq as i32,
            head_dim as i32,
            heads_per_kv as i32,
            scale,
            Device::Gpu,
        ),
        _ => rotor_flash_decode_sdpa::<4>(
            &q_arr,
            &codes,
            &scales,
            &norms,
            &rotors,
            &v_arr,
            mask_arr.as_ref(),
            b as i32,
            kv_h as i32,
            kv_seq as i32,
            head_dim as i32,
            heads_per_kv as i32,
            scale,
            Device::Gpu,
        ),
    }
    .expect("rotor_flash_decode_sdpa");

    let got = array_to_f32(&out);
    let want = ref_attention(
        &q,
        &k_deq,
        &v,
        mask_vec.as_deref(),
        n_q_heads,
        kv_h,
        kv_seq,
        head_dim,
        scale,
    );

    assert_eq!(got.len(), want.len(), "bits={bits}: output length");
    let mut max_err = 0.0_f32;
    for (g, w) in got.iter().zip(want.iter()) {
        max_err = max_err.max((g - w).abs());
    }
    // bf16 has ~8 mantissa bits (~3e-3 relative). The kernel accumulates in
    // f32 but is fed the same quantized codes as the oracle, so the only
    // divergence is summation order; this bound is well inside bf16 tolerance.
    assert!(
        max_err < 2e-3,
        "bits={bits} kv_h={kv_h} hpk={heads_per_kv} S={kv_seq} D={head_dim}: \
         max_err={max_err} exceeds bf16 tolerance"
    );
    true
}

// ── Oracle: kernel == CPU dequant reference ───────────────────────────────────

#[test]
fn rotor3_flash_decode_matches_cpu_dequant_reference() {
    // Bonsai / Qwen3 shape: head_dim 128, GQA 4:1.
    run_oracle(3, 2, 4, 40, 128);
}

#[test]
fn rotor4_flash_decode_matches_cpu_dequant_reference() {
    run_oracle(4, 2, 4, 40, 128);
}

#[test]
fn rotor3_flash_decode_matches_reference_across_tiles() {
    // kv_seq is set above TILE_SIZE so the P2 log-sum-exp merge runs over
    // several tiles rather than a single one.
    run_oracle(3, 1, 8, (TILE_SIZE as usize) * 2 + 22, 128);
}

#[test]
fn rotor4_flash_decode_matches_reference_across_tiles() {
    run_oracle(4, 1, 8, (TILE_SIZE as usize) * 2 + 22, 128);
}

#[test]
fn rotor3_flash_decode_matches_reference_head_dim_512() {
    // Gemma4 e2b/e4b global layers run head_dim = 512 with a single KV head.
    // head_dim % 3 != 0, so the last rotor group is tail-padded.
    run_oracle(3, 1, 8, 70, 512);
}

#[test]
fn rotor4_flash_decode_matches_reference_head_dim_512() {
    run_oracle(4, 1, 8, 70, 512);
}

#[test]
fn rotor3_flash_decode_matches_reference_head_dim_256() {
    run_oracle(3, 1, 4, 70, 256);
}

// ── Additive mask ─────────────────────────────────────────────────────────────
//
// Without these the kernel's mask read
// (`mask_flat[(b * n_q_heads + hq) * kv_seq + t]`) and the dispatcher's mask
// flatten are never executed — every other test passes `None`, so a transposed
// or mis-strided mask index would pass the whole suite.

#[test]
fn rotor3_flash_decode_matches_reference_with_additive_mask() {
    // kv_h=2, heads_per_kv=4 also pins the GQA (q_head -> kv_head) mapping: the
    // mask is indexed by q_head while K is indexed by kv_head, so conflating the
    // two shows up here.
    run_oracle_masked(3, 2, 4, 40, 128, true);
}

#[test]
fn rotor4_flash_decode_matches_reference_with_additive_mask() {
    run_oracle_masked(4, 2, 4, 40, 128, true);
}

#[test]
fn rotor3_flash_decode_matches_reference_with_mask_across_tiles() {
    // Mask + multi-tile: the per-tile online softmax and the P2 merge both have
    // to see the masked scores.
    run_oracle_masked(3, 2, 4, (TILE_SIZE as usize) * 2 + 22, 128, true);
}

// ── Gates ─────────────────────────────────────────────────────────────────────

#[test]
fn rotor_flash_decode_rejects_non_pow2_head_dim() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let err = rotor_flash_decode_sdpa::<3>(
        &dummy,
        &codes,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        None,
        1,
        1,
        8,
        96,
        1,
        1.0,
        Device::Gpu,
    );
    assert!(
        err.is_err(),
        "head_dim=96 is not a power of two — the tree reduction requires one"
    );
}

#[test]
fn rotor_flash_decode_rejects_head_dim_above_max() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let over = ROTOR_FLASH_HEAD_DIM_MAX * 2;
    let err = rotor_flash_decode_sdpa::<3>(
        &dummy,
        &codes,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        None,
        1,
        1,
        8,
        over,
        1,
        1.0,
        Device::Gpu,
    );
    assert!(
        err.is_err(),
        "head_dim={over} exceeds the static threadgroup-array ceiling"
    );
}

#[test]
fn rotor_flash_decode_rejects_unsupported_bits() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    // An unknown bit width must be an explicit error, never a silent fallback
    // to a kernel built for a different unpack width.
    let err = rotor_flash_decode_sdpa::<5>(
        &dummy,
        &codes,
        &dummy,
        &dummy,
        &dummy,
        &dummy,
        None,
        1,
        1,
        8,
        128,
        1,
        1.0,
        Device::Gpu,
    );
    assert!(err.is_err(), "BITS=5 must be rejected");
}

#[test]
fn build_rotor_flash_header_rejects_unsupported_bits() {
    for bits in [0_u8, 1, 2, 5, 8] {
        assert!(
            build_rotor_flash_header(bits).is_err(),
            "bits={bits} must be rejected, not silently mapped to a 3/4-bit header"
        );
    }
}

#[test]
fn build_rotor_flash_header_encodes_bit_width() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 3 and 4 are the supported widths; a failure here is the assertion"
    )]
    let h3 = build_rotor_flash_header(3).unwrap();
    #[allow(
        clippy::unwrap_used,
        reason = "bits 3 and 4 are the supported widths; a failure here is the assertion"
    )]
    let h4 = build_rotor_flash_header(4).unwrap();

    // The 3-bit and 4-bit variants MUST differ in unpack width — the shared
    // body reads RF_BITS / RF_MASK from here, so identical headers would mean
    // one width silently decoding with the other's stride.
    assert!(h3.contains("#define RF_BITS 3u"), "h3 RF_BITS");
    assert!(h3.contains("#define RF_MASK 0x7u"), "h3 RF_MASK");
    assert!(h3.contains("constant float RF_CB[8]"), "h3 codebook size");
    assert!(h4.contains("#define RF_BITS 4u"), "h4 RF_BITS");
    assert!(h4.contains("#define RF_MASK 0xFu"), "h4 RF_MASK");
    assert!(h4.contains("constant float RF_CB[16]"), "h4 codebook size");
    assert_ne!(h3, h4, "the two bit widths must not share a header");
}

#[test]
fn rotor_flash_header_exposes_reusable_decode_fn() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 3 is a supported width; a failure here is the assertion"
    )]
    let h = build_rotor_flash_header(3).unwrap();
    // The K-decode is emitted as a header function so a future quantized-V
    // flash kernel can call it instead of copying the sandwich.
    assert!(
        h.contains("inline float rf_decode_k_lane("),
        "header must expose the shared per-lane K decode as a function"
    );
}

// ── Dispatch counter ──────────────────────────────────────────────────────────

// The counter is process-global and `cargo test` runs this binary's tests on
// parallel threads, so only a **relative** delta around a known dispatch is
// assertable here — a concurrent test can inflate it at any time. An absolute
// "starts at zero" check would be a flake, and the negative case ("did not
// fire") is asserted on cache-local state in
// `kvcache::rotor_flash_dispatch_tests` instead.
#[test]
fn rotor_flash_decode_dispatch_count_increments_on_gpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    let before3 = rotor3_flash_decode_dispatch_count();
    let before4 = rotor4_flash_decode_dispatch_count();
    assert!(run_oracle(3, 1, 2, 8, 128));
    assert!(run_oracle(4, 1, 2, 8, 128));
    assert!(
        rotor3_flash_decode_dispatch_count() > before3,
        "rotor3 flash-decode kernel did not fire"
    );
    assert!(
        rotor4_flash_decode_dispatch_count() > before4,
        "rotor4 flash-decode kernel did not fire"
    );
}

// ── Probe-snapshot drift guards ───────────────────────────────────────────────

#[test]
fn hdr_probe_snapshot_b3_matches_builder() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 3 is a supported width; a failure here is the assertion"
    )]
    let built = build_rotor_flash_header(3).unwrap();
    assert_eq!(
        built,
        include_str!("metal/probes/rotor_flash_decode_p1_b3.hdr.metal"),
        "stale snapshot: refresh metal/probes/rotor_flash_decode_p1_b3.hdr.metal"
    );
}

#[test]
fn hdr_probe_snapshot_b4_matches_builder() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 4 is a supported width; a failure here is the assertion"
    )]
    let built = build_rotor_flash_header(4).unwrap();
    assert_eq!(
        built,
        include_str!("metal/probes/rotor_flash_decode_p1_b4.hdr.metal"),
        "stale snapshot: refresh metal/probes/rotor_flash_decode_p1_b4.hdr.metal"
    );
}
