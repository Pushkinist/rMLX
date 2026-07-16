//! GPU parity tests for the rotor symmetric (quant-K + quant-V) flash-decode
//! kernel (BITS in {3, 4}).
//!
//! The oracle is the codec's own CPU dequant
//! ([`crate::rotorquant::rotor3_decode`] / [`rotor4_decode`]) applied to **both**
//! axes and fed through a scalar reference attention chain. Decoding V with the
//! codec's own reference — rather than comparing against the bf16 mirror this
//! kernel deletes — is the point: it proves the in-kernel V unpack reproduces
//! the codec, not that it agrees with a buffer that no longer exists.

use super::*;
use crate::clifford::make_rotor_table;
use crate::rotor_flash_decode_msl::build_rotor_flash_header;
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

/// Scalar reference attention over already-dequantized K **and** V.
///
/// Both `k_deq` and `v_deq` are sequence-major (`[(s * kv_h + h), head_dim]`) —
/// the layout both rotor rings accumulate and the kernel reads. This is the
/// difference from the bf16-V sibling's reference, whose V was head-major.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "test reference: all indices derived from the shape params under test"
)]
fn ref_attention_quant_v(
    q: &[f32],
    k_deq: &[f32],
    v_deq: &[f32],
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

        // Weighted V sum — V read sequence-major, same as K.
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for (s, &p) in scores.iter().enumerate() {
                acc += p * v_deq[(s * kv_h + h) * head_dim + d];
            }
            out[hq * head_dim + d] = acc / denom;
        }
    }
    out
}

/// One axis's packed rotor buffers plus the matching CPU dequant oracle.
struct EncodedAxis {
    codes: Array,
    scales: Array,
    norms: Array,
    rotors: Array,
    dequant: Vec<f32>,
}

impl EncodedAxis {
    fn as_packed(&self) -> RotorPackedAxis<'_> {
        RotorPackedAxis {
            codes: &self.codes,
            scales: &self.scales,
            norms: &self.norms,
            rotors: &self.rotors,
        }
    }
}

/// Encode `seq_major` with the rotor codec on CPU and return the GPU-resident
/// packed buffers plus the CPU dequant oracle.
///
/// `layer_idx` seeds the rotor table: the K and V stores of a real cache each
/// build their own, so the tests feed distinct tables to prove the kernel reads
/// each axis against its own rotors rather than reusing K's for V.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor_encode_axis(seq_major: &[f32], head_dim: usize, bits: u8, layer_idx: u32) -> EncodedAxis {
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(layer_idx, 0, n_groups);

    let (codes, scales, norms) = if bits == 3 {
        rotor3_encode(seq_major, &rotors, head_dim).expect("rotor3_encode")
    } else {
        rotor4_encode(seq_major, &rotors, head_dim).expect("rotor4_encode")
    };
    let dequant = if bits == 3 {
        rotor3_decode(&codes, &scales, &norms, &rotors, head_dim).expect("rotor3_decode")
    } else {
        rotor4_decode(&codes, &scales, &norms, &rotors, head_dim).expect("rotor4_decode")
    };

    EncodedAxis {
        codes: make_u32_array(&codes, &[codes.len() as i32]),
        scales: make_f32_array(&scales, &[scales.len() as i32]),
        norms: make_f32_array(&norms, &[norms.len() as i32]),
        rotors: make_f32_array(&rotors, &[rotors.len() as i32]),
        dequant,
    }
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
    // K and V are both generated in the stores' sequence-major order: token
    // (s, h) at row (s * kv_h + h). Different seeds so a K/V buffer swap in the
    // dispatcher cannot pass.
    let k = lcg_data(n_tokens * head_dim, 0xB0B);
    let v = lcg_data(n_tokens * head_dim, 0xC0FFEE);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Distinct layer_idx per axis => distinct rotor tables, as a real cache has.
    let k_axis = rotor_encode_axis(&k, head_dim, bits, 0);
    let v_axis = rotor_encode_axis(&v, head_dim, bits, 1);
    let q_arr = make_f32_array(&q, &[b as i32, n_q_heads as i32, 1, head_dim as i32]);

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

    let shape = RotorFlashShape {
        b: b as i32,
        kv_h: kv_h as i32,
        kv_seq: kv_seq as i32,
        head_dim: head_dim as i32,
        heads_per_kv: heads_per_kv as i32,
    };

    let out = match bits {
        3 => rotor_flash_decode_symv_sdpa::<3>(
            &q_arr,
            k_axis.as_packed(),
            v_axis.as_packed(),
            mask_arr.as_ref(),
            shape,
            scale,
            Device::Gpu,
        ),
        _ => rotor_flash_decode_symv_sdpa::<4>(
            &q_arr,
            k_axis.as_packed(),
            v_axis.as_packed(),
            mask_arr.as_ref(),
            shape,
            scale,
            Device::Gpu,
        ),
    }
    .expect("rotor_flash_decode_symv_sdpa");

    let got = array_to_f32(&out);
    let want = ref_attention_quant_v(
        &q,
        &k_axis.dequant,
        &v_axis.dequant,
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

// ── Oracle: kernel == CPU dequant reference on BOTH axes ──────────────────────

#[test]
fn rotor3_symv_flash_decode_matches_cpu_dequant_reference() {
    // Bonsai / Qwen3 shape: head_dim 128, GQA 4:1.
    run_oracle(3, 2, 4, 40, 128);
}

#[test]
fn rotor4_symv_flash_decode_matches_cpu_dequant_reference() {
    run_oracle(4, 2, 4, 40, 128);
}

#[test]
fn rotor3_symv_flash_decode_matches_reference_across_tiles() {
    // kv_seq is set above TILE_SIZE so the P2 log-sum-exp merge runs over
    // several tiles rather than a single one.
    run_oracle(3, 1, 8, (TILE_SIZE as usize) * 2 + 22, 128);
}

#[test]
fn rotor4_symv_flash_decode_matches_reference_across_tiles() {
    run_oracle(4, 1, 8, (TILE_SIZE as usize) * 2 + 22, 128);
}

#[test]
fn rotor3_symv_flash_decode_matches_reference_head_dim_512() {
    // Gemma4 e2b/e4b global layers run head_dim = 512 with a single KV head.
    // head_dim % 3 != 0, so the last rotor group is tail-padded.
    run_oracle(3, 1, 8, 70, 512);
}

#[test]
fn rotor4_symv_flash_decode_matches_reference_head_dim_512() {
    run_oracle(4, 1, 8, 70, 512);
}

#[test]
fn rotor3_symv_flash_decode_matches_reference_head_dim_256() {
    run_oracle(3, 1, 4, 70, 256);
}

#[test]
fn rotor4_symv_flash_decode_matches_reference_head_dim_64() {
    // Smallest production-plausible head_dim; head_dim % 3 != 0 again, and a
    // single rotor-group tail lane.
    run_oracle(4, 2, 2, 40, 64);
}

// ── Additive mask + GQA ───────────────────────────────────────────────────────
//
// Without these the kernel's mask read
// (`mask_flat[(b * n_q_heads + hq) * kv_seq + t]`) and the dispatcher's mask
// flatten are never executed — every other test passes `None`, so a transposed
// or mis-strided mask index would pass the whole suite.

#[test]
fn rotor3_symv_flash_decode_matches_reference_with_additive_mask() {
    // kv_h=2, heads_per_kv=4 also pins the GQA (q_head -> kv_head) mapping: the
    // mask is indexed by q_head while K/V are indexed by kv_head, so conflating
    // the two shows up here.
    run_oracle_masked(3, 2, 4, 40, 128, true);
}

#[test]
fn rotor4_symv_flash_decode_matches_reference_with_additive_mask() {
    run_oracle_masked(4, 2, 4, 40, 128, true);
}

#[test]
fn rotor3_symv_flash_decode_matches_reference_with_mask_across_tiles() {
    // Mask + multi-tile: the per-tile online softmax and the P2 merge both have
    // to see the masked scores.
    run_oracle_masked(3, 2, 4, (TILE_SIZE as usize) * 2 + 22, 128, true);
}

#[test]
fn rotor4_symv_flash_decode_matches_reference_with_mask_high_gqa() {
    // kv_h=2, heads_per_kv=8: a wider GQA fan-out than the shapes above, so a
    // `hq / heads_per_kv` vs `hq % kv_h` mix-up in the kv-head mapping lands on
    // a different KV row and fails here.
    run_oracle_masked(4, 2, 8, 40, 128, true);
}

// ── Gates ─────────────────────────────────────────────────────────────────────
//
// Every gate below is a refusal, not a fallback: an unsupported shape or bit
// width must error rather than decode against another codec's kernel.

/// A one-element dummy axis for the shape-rejection gates, which reject before
/// any buffer is read. Borrows the caller's locals — the gates never look at
/// the contents, only at the shape metadata.
fn dummy_axis<'a>(codes: &'a Array, f: &'a Array) -> RotorPackedAxis<'a> {
    RotorPackedAxis {
        codes,
        scales: f,
        norms: f,
        rotors: f,
    }
}

#[test]
fn rotor_symv_flash_decode_rejects_non_pow2_head_dim() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let axis = dummy_axis(&codes, &dummy);
    let err = rotor_flash_decode_symv_sdpa::<3>(
        &dummy,
        axis,
        axis,
        None,
        RotorFlashShape {
            b: 1,
            kv_h: 1,
            kv_seq: 8,
            head_dim: 96,
            heads_per_kv: 1,
        },
        1.0,
        Device::Gpu,
    );
    assert!(
        err.is_err(),
        "head_dim=96 is not a power of two — the tree reduction requires one"
    );
}

#[test]
fn rotor_symv_flash_decode_rejects_head_dim_above_max() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let axis = dummy_axis(&codes, &dummy);
    let over = ROTOR_FLASH_HEAD_DIM_MAX * 2;
    let err = rotor_flash_decode_symv_sdpa::<3>(
        &dummy,
        axis,
        axis,
        None,
        RotorFlashShape {
            b: 1,
            kv_h: 1,
            kv_seq: 8,
            head_dim: over,
            heads_per_kv: 1,
        },
        1.0,
        Device::Gpu,
    );
    assert!(
        err.is_err(),
        "head_dim={over} exceeds the static threadgroup-array ceiling"
    );
}

#[test]
fn rotor_symv_flash_decode_rejects_unsupported_bits() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let axis = dummy_axis(&codes, &dummy);
    // An unknown bit width must be an explicit error, never a silent fallback
    // to a kernel built for a different unpack width.
    let err = rotor_flash_decode_symv_sdpa::<5>(
        &dummy,
        axis,
        axis,
        None,
        RotorFlashShape {
            b: 1,
            kv_h: 1,
            kv_seq: 8,
            head_dim: 128,
            heads_per_kv: 1,
        },
        1.0,
        Device::Gpu,
    );
    assert!(err.is_err(), "BITS=5 must be rejected");
}

#[test]
fn rotor_symv_flash_decode_rejects_non_positive_shapes() {
    if skip_if_no_gpu_env() {
        return;
    }
    let dummy = make_f32_array(&[0.0], &[1]);
    let codes = make_u32_array(&[0], &[1]);
    let axis = dummy_axis(&codes, &dummy);
    for (b, kv_h, kv_seq, heads_per_kv) in [(0, 1, 8, 1), (1, 0, 8, 1), (1, 1, 0, 1), (1, 1, 8, 0)]
    {
        let err = rotor_flash_decode_symv_sdpa::<3>(
            &dummy,
            axis,
            axis,
            None,
            RotorFlashShape {
                b,
                kv_h,
                kv_seq,
                head_dim: 128,
                heads_per_kv,
            },
            1.0,
            Device::Gpu,
        );
        assert!(
            err.is_err(),
            "b={b} kv_h={kv_h} kv_seq={kv_seq} heads_per_kv={heads_per_kv} must be rejected"
        );
    }
}

// ── Header reuse ──────────────────────────────────────────────────────────────

#[test]
fn symv_kernel_reuses_the_shared_rotor_decode_fn() {
    #[allow(
        clippy::unwrap_used,
        reason = "bits 3 is a supported width; a failure here is the assertion"
    )]
    let h = build_rotor_flash_header(3).unwrap();
    // This kernel's whole design rests on calling the sibling's per-lane rotor
    // decode unchanged, for both axes. If the header stops exposing it as a
    // function, the body stops compiling — pin the contract here so the reason
    // is legible rather than an MSL build error at first dispatch.
    assert!(
        h.contains("inline float rf_decode_k_lane("),
        "header must expose the shared per-lane rotor decode as a function"
    );
    // Both axes are unpacked by the same body, so the body must call it twice —
    // once per axis. A single call would mean one axis is being read some other
    // way (or not at all).
    let body = include_str!("metal/rotor_flash_decode_symv_p1.metal");
    assert_eq!(
        body.matches("rf_decode_k_lane(").count(),
        2,
        "symv body must decode exactly two axes through the shared lane decode"
    );
    assert!(
        body.contains("rf_decode_k_lane(k_codes, k_scales, k_norms, k_rotors,"),
        "K axis must be read from the K ring"
    );
    assert!(
        body.contains("rf_decode_k_lane(v_codes, v_scales, v_norms, v_rotors,"),
        "V axis must be read from the V ring, with the V store's own rotor table"
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
fn rotor_symv_flash_decode_dispatch_count_increments_on_gpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    let before3 = rotor3_symv_flash_decode_dispatch_count();
    let before4 = rotor4_symv_flash_decode_dispatch_count();
    assert!(run_oracle(3, 1, 2, 8, 128));
    assert!(run_oracle(4, 1, 2, 8, 128));
    assert!(
        rotor3_symv_flash_decode_dispatch_count() > before3,
        "rotor3 symv flash-decode kernel did not fire"
    );
    assert!(
        rotor4_symv_flash_decode_dispatch_count() > before4,
        "rotor4 symv flash-decode kernel did not fire"
    );
}
