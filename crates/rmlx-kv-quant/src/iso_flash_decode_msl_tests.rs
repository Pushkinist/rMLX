//! GPU parity tests for the iso flash-decode kernel (BITS in {3, 4}).
//!
//! The oracle is the codec's own CPU dequant
//! ([`crate::isoquant::iso_decode_fast`]) fed through a scalar reference
//! attention chain. That is exactly what the production path did before this
//! kernel existed, so a match proves the kernel reproduces the path it
//! replaces — not merely that it is self-consistent.

use super::*;
use crate::isoquant::{iso_decode_fast, iso_encode_fast};
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
/// iso ring accumulates and the kernel reads. `v` is head-major
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

/// Encode `k_seq_major` with the iso codec on CPU and return the GPU-resident
/// packed buffers plus the CPU dequant oracle.
///
/// `norms` is deduplicated to per-token here: `iso_encode_fast` already emits
/// one norm per token, which is exactly what the ring and the kernel want.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn iso_encode_for_test(
    k_seq_major: &[f32],
    head_dim: usize,
    bits: u8,
) -> (Array, Array, Array, Vec<f32>) {
    let (codes, scales, quaternions, norms) =
        iso_encode_fast(k_seq_major, head_dim, ISO_K3_GROUP_SIZE, bits).expect("iso_encode_fast");
    let dequant = iso_decode_fast(
        &codes,
        &scales,
        &quaternions,
        &norms,
        head_dim,
        ISO_K3_GROUP_SIZE,
        bits,
    )
    .expect("iso_decode_fast");

    (
        make_u32_array(&codes, &[codes.len() as i32]),
        make_f32_array(&scales, &[scales.len() as i32]),
        make_f32_array(&norms, &[norms.len() as i32]),
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

    let (codes, scales, norms, k_deq) = iso_encode_for_test(&k, head_dim, bits);
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
        3 => iso_flash_decode_sdpa::<3>(
            &q_arr,
            &codes,
            &scales,
            &norms,
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
        _ => iso_flash_decode_sdpa::<4>(
            &q_arr,
            &codes,
            &scales,
            &norms,
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
    .expect("iso_flash_decode_sdpa");

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
fn iso3_flash_decode_matches_cpu_dequant_reference() {
    // Bonsai / Qwen3 shape: head_dim 128, GQA 4:1.
    run_oracle(3, 2, 4, 40, 128);
}

#[test]
fn iso4_flash_decode_matches_cpu_dequant_reference() {
    run_oracle(4, 2, 4, 40, 128);
}

#[test]
fn iso3_flash_decode_matches_reference_across_tiles() {
    // kv_seq is set above TILE_SIZE so the P2 log-sum-exp merge runs over
    // several tiles rather than a single one.
    run_oracle(3, 1, 8, (TILE_SIZE as usize) * 2 + 22, 128);
}

#[test]
fn iso4_flash_decode_matches_reference_across_tiles() {
    run_oracle(4, 1, 8, (TILE_SIZE as usize) * 2 + 22, 128);
}

#[test]
fn iso3_flash_decode_matches_reference_head_dim_512() {
    // Gemma4 e2b/e4b global layers run head_dim = 512 with a single KV head.
    run_oracle(3, 1, 8, 70, 512);
}

#[test]
fn iso4_flash_decode_matches_reference_head_dim_512() {
    run_oracle(4, 1, 8, 70, 512);
}

#[test]
fn iso3_flash_decode_matches_reference_head_dim_256() {
    // medgemma / Gemma3 shape.
    run_oracle(3, 1, 4, 70, 256);
}

#[test]
fn iso4_flash_decode_matches_reference_head_dim_256() {
    run_oracle(4, 1, 4, 70, 256);
}

// ── Additive mask ─────────────────────────────────────────────────────────────
//
// Without these the kernel's mask read
// (`mask_flat[(b * n_q_heads + hq) * kv_seq + t]`) and the dispatcher's mask
// flatten are never executed — every other test passes `None`, so a transposed
// or mis-strided mask index would pass the whole suite.

#[test]
fn iso3_flash_decode_matches_reference_with_additive_mask() {
    // kv_h=2, heads_per_kv=4 also pins the GQA (q_head -> kv_head) mapping: the
    // mask is indexed by q_head while K is indexed by kv_head, so conflating the
    // two shows up here.
    run_oracle_masked(3, 2, 4, 40, 128, true);
}

#[test]
fn iso4_flash_decode_matches_reference_with_additive_mask() {
    run_oracle_masked(4, 2, 4, 40, 128, true);
}

#[test]
fn iso3_flash_decode_matches_reference_with_mask_across_tiles() {
    // Mask + multi-tile: the per-tile online softmax and the P2 merge both have
    // to see the masked scores.
    run_oracle_masked(3, 2, 4, (TILE_SIZE as usize) * 2 + 22, 128, true);
}

#[test]
fn iso4_flash_decode_matches_reference_with_mask_across_tiles() {
    run_oracle_masked(4, 2, 4, (TILE_SIZE as usize) * 2 + 22, 128, true);
}

// ── GQA ───────────────────────────────────────────────────────────────────────

#[test]
fn iso3_flash_decode_matches_reference_multi_kv_head_gqa() {
    // kv_h=4 with heads_per_kv=2: several KV heads AND a GQA fan-out, so a
    // kv_h_idx derived with the wrong divisor reads another head's K.
    run_oracle(3, 4, 2, 40, 128);
}

#[test]
fn iso3_flash_decode_matches_reference_mha_no_gqa() {
    // heads_per_kv=1 (plain MHA): kv_h_idx == hq, the degenerate GQA case.
    run_oracle(3, 4, 1, 40, 128);
}

// ── Gates ─────────────────────────────────────────────────────────────────────

#[test]
fn iso_flash_decode_rejects_non_pow2_head_dim() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    // head_dim=96 is a multiple of the quaternion block size but not a power of
    // two, so it isolates the tree-reduction gate from the group-size gate.
    let err = iso_flash_decode_sdpa::<3>(
        &dummy,
        &codes,
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
fn iso_flash_decode_rejects_head_dim_above_max() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let err = iso_flash_decode_sdpa::<3>(
        &dummy,
        &codes,
        &dummy,
        &dummy,
        &dummy,
        None,
        1,
        1,
        8,
        ISO_FLASH_HEAD_DIM_MAX * 2,
        1,
        1.0,
        Device::Gpu,
    );
    assert!(
        err.is_err(),
        "head_dim above ISO_FLASH_HEAD_DIM_MAX must error — the threadgroup arrays are \
         statically sized to it"
    );
}

#[test]
fn iso_flash_decode_rejects_unsupported_bits() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    // BITS=5 must not silently decode as 3-bit or 4-bit.
    let err = iso_flash_decode_sdpa::<5>(
        &dummy,
        &codes,
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
    assert!(
        err.is_err(),
        "BITS=5 has no codebook — it must error, never fall back to another width"
    );
}

#[test]
fn iso_flash_header_rejects_unsupported_bits() {
    // Header build is the other place a bad width could silently pick a
    // codebook. No GPU needed.
    assert!(build_iso_flash_header(3).is_ok(), "bits=3 is supported");
    assert!(build_iso_flash_header(4).is_ok(), "bits=4 is supported");
    for bits in [0_u8, 1, 2, 5, 8] {
        assert!(
            build_iso_flash_header(bits).is_err(),
            "bits={bits} must be rejected, not rendered with a wrong-width codebook"
        );
    }
}

#[test]
fn iso_flash_header_carries_the_matching_codebook_width() {
    #[allow(
        clippy::expect_used,
        reason = "test: bits 3/4 are supported, asserted directly above"
    )]
    let h3 = build_iso_flash_header(3).expect("header bits=3");
    #[allow(
        clippy::expect_used,
        reason = "test: bits 3/4 are supported, asserted directly above"
    )]
    let h4 = build_iso_flash_header(4).expect("header bits=4");
    assert!(h3.contains("#define IF_BITS 3u"), "bits=3 unpack width");
    assert!(h3.contains("#define IF_MASK 0x7u"), "bits=3 unpack mask");
    assert!(h3.contains("constant float ISO_CB[8]"), "bits=3 codebook");
    assert!(h4.contains("#define IF_BITS 4u"), "bits=4 unpack width");
    assert!(h4.contains("#define IF_MASK 0xFu"), "bits=4 unpack mask");
    assert!(h4.contains("constant float ISO_CB[16]"), "bits=4 codebook");
    // The shared K-decode is what a quantized-V flash kernel calls; it has to
    // be a header function, not inlined into the body.
    assert!(
        h3.contains("inline float if_decode_k_lane("),
        "the reusable per-lane decode must live in the header"
    );
}

// ── Fixed-quaternion coupling ─────────────────────────────────────────────────

#[test]
fn assert_fixed_quat_accepts_the_encoder_table() {
    // What `iso_encode_fast` actually writes must pass — otherwise the guard
    // would reject every real store.
    let k = lcg_data(4 * 128, 0xD00D);
    #[allow(clippy::expect_used, reason = "test: fixed shape known-valid")]
    let (_c, _s, quaternions, _n) =
        iso_encode_fast(&k, 128, ISO_K3_GROUP_SIZE, 3).expect("iso_encode_fast");
    assert!(
        assert_fixed_quat_blocks(&quaternions, "test").is_ok(),
        "the encoder's own quaternion table must satisfy the kernel's fixed-quat contract"
    );
}

#[test]
fn assert_fixed_quat_rejects_a_per_group_table() {
    // The failure this guard exists for: a store whose groups carry their own
    // quaternions decodes to garbage through a header that baked in a constant.
    let mut quats: Vec<f32> = FIXED_QUAT.to_vec();
    quats.extend_from_slice(&[0.5, 0.5, 0.5, 0.5]);
    let err = assert_fixed_quat_blocks(&quats, "test");
    assert!(
        err.is_err(),
        "a group with a non-fixed quaternion must be rejected, not decoded against the \
         header's constant"
    );
}

#[test]
fn assert_fixed_quat_rejects_a_ragged_table() {
    let err = assert_fixed_quat_blocks(&[1.0, 0.0, 0.0], "test");
    assert!(
        err.is_err(),
        "a quaternion table that is not a multiple of 4 is malformed"
    );
}

// ── Probe-snapshot drift guards ───────────────────────────────────────────────
//
// The `metal/probes/*.hdr.metal` files are captured output of the builder
// below, compiled by `make check-metal-compiles`. Without these the snapshot
// could drift from what production actually dispatches, and the compile gate
// would happily keep verifying the stale copy.

#[test]
fn hdr_probe_snapshot_b3_matches_builder() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 3 is a supported width; a failure here is the assertion"
    )]
    let built = build_iso_flash_header(3).unwrap();
    assert_eq!(
        built,
        include_str!("metal/probes/iso_flash_decode_p1_b3.hdr.metal"),
        "stale snapshot: refresh metal/probes/iso_flash_decode_p1_b3.hdr.metal"
    );
}

#[test]
fn hdr_probe_snapshot_b4_matches_builder() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 4 is a supported width; a failure here is the assertion"
    )]
    let built = build_iso_flash_header(4).unwrap();
    assert_eq!(
        built,
        include_str!("metal/probes/iso_flash_decode_p1_b4.hdr.metal"),
        "stale snapshot: refresh metal/probes/iso_flash_decode_p1_b4.hdr.metal"
    );
}
