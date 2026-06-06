use super::*;
use crate::bf16::bf16_to_f32;

// Helper: encode an f32 as bf16 LE bytes.
fn f32_to_bf16_le(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let bf16 = (bits >> 16) as u16;
    bf16.to_le_bytes()
}

// Helper: build scales/biases buffers for a constant scale and bias.
fn const_sb(rows: usize, groups_per_row: usize, val: f32) -> Vec<u8> {
    let le = f32_to_bf16_le(val);
    let n = rows * groups_per_row;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        out.push(le[0]);
        out.push(le[1]);
    }
    out
}

// ── Bias sign convention ─────────────────────────────────────────────────

/// Documents the bias sign convention used by rMLX: ADDITIVE.
///
/// Ground truth: docs/03-mlx-safetensors-format.md §Affine:
/// `w_fp = s * x_q + b`
///
/// With scale=1.0, code=3, bias=0.5 → expected 3.5.
/// If the convention were subtractive (s*x_q - b), result would be 2.5.
#[test]
fn bias_convention_is_additive() {
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    // All codes = 3.
    let codes = vec![3u32; 32];
    let packed = pack_codes_for_test(&codes, &params);

    let scale_val = 1.0_f32;
    let bias_val = 0.5_f32;
    let scales = const_sb(1, 1, scale_val);
    let biases = const_sb(1, 1, bias_val);

    let out = dequant_vec(&params, &packed, &scales, &biases).unwrap();
    // Additive: 1.0 * 3 + 0.5 = 3.5
    for &v in &out {
        assert!(
            (v - 3.5_f32).abs() < 1e-5,
            "expected 3.5 (additive bias), got {v}"
        );
    }
}

// ── Zero-error case: scale=1, bias=0, codes=0..2^bits-1 cycling ─────────

fn zero_error_test(bits: u8) {
    let group_size = 32u32;
    let cols = 64usize; // 2 groups
    let rows = 2usize;
    let max_code = (1u32 << bits) - 1;

    let params = AffineParams {
        bits,
        group_size,
        storage: CodeStorage::U32Le,
        rows,
        cols,
    };

    // Codes cycle 0..max_code.
    let codes: Vec<u32> = (0..rows * cols).map(|i| (i as u32) & max_code).collect();
    let packed = pack_codes_for_test(&codes, &params);

    let groups_per_row = cols / group_size as usize;
    let scales = const_sb(rows, groups_per_row, 1.0);
    let biases = const_sb(rows, groups_per_row, 0.0);

    let out = dequant_vec(&params, &packed, &scales, &biases).unwrap();

    for (i, (&code, &val)) in codes.iter().zip(out.iter()).enumerate() {
        assert!(
            (val - code as f32).abs() < 1e-5,
            "bits={bits} idx={i}: code={code} expected {}, got {val}",
            code as f32
        );
    }
}

#[test]
fn zero_error_bits2() {
    zero_error_test(2);
}
#[test]
fn zero_error_bits3() {
    zero_error_test(3);
}
#[test]
fn zero_error_bits4() {
    zero_error_test(4);
}
#[test]
fn zero_error_bits5() {
    zero_error_test(5);
}
#[test]
fn zero_error_bits6() {
    zero_error_test(6);
}
#[test]
fn zero_error_bits8() {
    zero_error_test(8);
}

// ── Round-trip: quantize→pack→dequant, error ≤ scale/2 ─────────────────

fn round_trip_test(bits: u8, group_size: u32, storage: CodeStorage) {
    // Use deterministic "random" weights via a simple LCG.
    let rows = 4usize;
    let cols = group_size as usize * 2; // 2 groups per row
    let max_code = (1u32 << bits) - 1;
    let max_code_f = max_code as f32;

    // Generate weights in a reasonable range.
    let mut state = 0x12345678u64;
    let weights: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32); // [0,1)
            frac * 2.0 - 1.0 // [-1, 1)
        })
        .collect();

    // Compute per-group scale and bias (zero-point), quantize.
    let groups_per_row = cols / group_size as usize;
    let mut scales_f32 = vec![0.0_f32; rows * groups_per_row];
    let mut biases_f32 = vec![0.0_f32; rows * groups_per_row];
    let mut codes = vec![0u32; rows * cols];

    for r in 0..rows {
        for g in 0..groups_per_row {
            let start = r * cols + g * group_size as usize;
            let end = start + group_size as usize;
            let group = &weights[start..end];

            let w_min = group.iter().copied().fold(f32::INFINITY, f32::min);
            let w_max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let range = (w_max - w_min).max(1e-8);
            let scale = range / max_code_f;
            // bias = w_min (so w ≈ scale * code + bias with code=0 → w_min)
            let bias = w_min;

            scales_f32[r * groups_per_row + g] = scale;
            biases_f32[r * groups_per_row + g] = bias;

            for c_local in 0..group_size as usize {
                let c = g * group_size as usize + c_local;
                let w = weights[r * cols + c];
                let q = ((w - bias) / scale).round() as i64;
                let q_clipped = q.clamp(0, i64::from(max_code)) as u32;
                codes[r * cols + c] = q_clipped;
            }
        }
    }

    let params = AffineParams {
        bits,
        group_size,
        storage,
        rows,
        cols,
    };
    let packed = pack_codes_for_test(&codes, &params);

    // Encode scales/biases as bf16.
    let scales_bf16: Vec<u8> = scales_f32.iter().flat_map(|&v| f32_to_bf16_le(v)).collect();
    let biases_bf16: Vec<u8> = biases_f32.iter().flat_map(|&v| f32_to_bf16_le(v)).collect();

    let out = dequant_vec(&params, &packed, &scales_bf16, &biases_bf16).unwrap();

    // Verify two things:
    //
    // 1. Correctness: dequant output matches the closed-form computation using
    // the *same bf16-rounded* scale and bias (bit-exact equality).
    //
    // 2. Reconstruction bound: the error from the *original* weight is bounded
    // by `scale_orig / 2` PLUS the bf16 rounding error on scale/bias.
    // Since bf16 has 7 mantissa bits (relative error ≤ 2^-7), and bias is
    // stored in the same dtype, the total extra error per element is at most
    // `|Δscale| * max_code + |Δbias|`. We bound this loosely as
    // `scale_orig * (1 / 64) * max_code + |Δbias|`.
    for r in 0..rows {
        for g in 0..groups_per_row {
            let sb_idx = r * groups_per_row + g;
            // Decode bf16 scale and bias exactly as dequant did.
            let scale_le = f32_to_bf16_le(scales_f32[sb_idx]);
            let bias_le = f32_to_bf16_le(biases_f32[sb_idx]);
            let scale_bf16 = bf16_to_f32(scale_le);
            let bias_bf16 = bf16_to_f32(bias_le);
            let scale_orig = scales_f32[sb_idx];
            let bias_orig = biases_f32[sb_idx];

            for c_local in 0..group_size as usize {
                let c = g * group_size as usize + c_local;
                let w_orig = weights[r * cols + c];
                let code = codes[r * cols + c];

                // 1. Bit-exact check: dequant output == manual computation.
                let w_manual = scale_bf16 * (code as f32) + bias_bf16;
                let w_dq = out[r * cols + c];
                assert!(
                    (w_dq - w_manual).abs() < 1e-6,
                    "bits={bits} gs={group_size} r={r} c={c}: \
                     dequant={w_dq:.8} != manual={w_manual:.8}"
                );

                // 2. Reconstruction bound from original weight.
                // quant step error ≤ scale_orig / 2
                // + bf16 error on scale × code
                // + bf16 error on bias
                let delta_scale = (scale_orig - scale_bf16).abs();
                let delta_bias = (bias_orig - bias_bf16).abs();
                let tolerance = scale_orig / 2.0 + delta_scale * (code as f32) + delta_bias + 1e-6; // float arithmetic floor

                let err = (w_orig - w_dq).abs();
                assert!(
                    err <= tolerance,
                    "bits={bits} gs={group_size} storage={storage:?} \
                     r={r} c={c}: err={err:.6} > tolerance={tolerance:.6} \
                     (w_orig={w_orig:.6}, w_dq={w_dq:.6}, code={code}, \
                     scale_orig={scale_orig:.8}, scale_bf16={scale_bf16:.8}, \
                     bias_orig={bias_orig:.8}, bias_bf16={bias_bf16:.8})"
                );
            }
        }
    }
}

#[test]
fn round_trip_4bit_gs32_u32le() {
    round_trip_test(4, 32, CodeStorage::U32Le);
}
#[test]
fn round_trip_4bit_gs64_u32le() {
    round_trip_test(4, 64, CodeStorage::U32Le);
}
#[test]
fn round_trip_4bit_gs128_u32le() {
    round_trip_test(4, 128, CodeStorage::U32Le);
}
#[test]
fn round_trip_8bit_gs32_u32le() {
    round_trip_test(8, 32, CodeStorage::U32Le);
}
#[test]
fn round_trip_8bit_gs64_u32le() {
    round_trip_test(8, 64, CodeStorage::U32Le);
}
#[test]
fn round_trip_8bit_gs128_u32le() {
    round_trip_test(8, 128, CodeStorage::U32Le);
}

// ── U32Le storage variant explicit test ──────────────────────────────────

#[test]
fn u32le_storage_variant() {
    // Pack 4-bit codes [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,...] via U32Le.
    // 32 codes per row → 4 u32 words per row.
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    let codes: Vec<u32> = (0..32u32).map(|i| i & 0xF).collect();
    let packed = pack_codes_for_test(&codes, &params);
    // Verify packed length = 32/8 * 4 = 16 bytes (8 codes per u32 → 4 u32 = 16 bytes).
    assert_eq!(packed.len(), 16);

    let scales = const_sb(1, 1, 1.0);
    let biases = const_sb(1, 1, 0.0);
    let out = dequant_vec(&params, &packed, &scales, &biases).unwrap();

    for (i, (&expected, &got)) in codes.iter().zip(out.iter()).enumerate() {
        assert!(
            (got - expected as f32).abs() < 1e-5,
            "u32le idx={i}: expected {expected}, got {got}"
        );
    }
}

// ── Shape error tests ────────────────────────────────────────────────────

#[test]
fn err_bits_unsupported() {
    let params = AffineParams {
        bits: 7,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    let result = dequant_vec(&params, &[], &[], &[]);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for bits=7"
    );
}

#[test]
fn err_group_size_unsupported() {
    let params = AffineParams {
        bits: 4,
        group_size: 16,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    let result = dequant_vec(&params, &[], &[], &[]);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for group_size=16"
    );
}

#[test]
fn err_cols_not_multiple_of_group_size() {
    let params = AffineParams {
        bits: 4,
        group_size: 128,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 129,
    };
    let result = dequant_vec(&params, &[], &[], &[]);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for cols=129 not multiple of group_size=128"
    );
}

#[test]
fn err_packed_codes_wrong_length() {
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    // Correct packed len = 16 bytes; provide 10 (wrong).
    let scales = const_sb(1, 1, 1.0);
    let biases = const_sb(1, 1, 0.0);
    let result = dequant_vec(&params, &[0u8; 10], &scales, &biases);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for wrong packed_codes length"
    );
}

#[test]
fn err_scales_wrong_length() {
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    let packed = pack_codes_for_test(&[0u32; 32], &params);
    // Correct scales len = 2 bytes; provide 4 (wrong).
    let biases = const_sb(1, 1, 0.0);
    let result = dequant_vec(&params, &packed, &[0u8; 4], &biases);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for wrong scales length"
    );
}

#[test]
fn err_biases_wrong_length() {
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    let packed = pack_codes_for_test(&[0u32; 32], &params);
    let scales = const_sb(1, 1, 1.0);
    // Correct biases len = 2 bytes; provide 4 (wrong).
    let result = dequant_vec(&params, &packed, &scales, &[0u8; 4]);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for wrong biases length"
    );
}

// ── U8 storage round-trip ────────────────────────────────────────────────

#[test]
fn u8_storage_4bit_round_trip() {
    // Verify U8 storage packs and unpacks correctly for 4-bit codes.
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U8,
        rows: 1,
        cols: 32,
    };
    let codes: Vec<u32> = (0..32u32).map(|i| i & 0xF).collect();
    let packed = pack_codes_for_test(&codes, &params);
    // 32 * 4 bits = 128 bits = 16 bytes.
    assert_eq!(packed.len(), 16);

    let scales = const_sb(1, 1, 1.0);
    let biases = const_sb(1, 1, 0.0);
    let out = dequant_vec(&params, &packed, &scales, &biases).unwrap();

    for (i, (&expected, &got)) in codes.iter().zip(out.iter()).enumerate() {
        assert!(
            (got - expected as f32).abs() < 1e-5,
            "u8 idx={i}: expected {expected}, got {got}"
        );
    }
}

// ── Verify bf16 round-trip in dequant pipeline ───────────────────────────

#[test]
fn bf16_scale_bias_round_trip_in_dequant() {
    // scale=2.0 (bf16 0x4000 = [0x00, 0x40]), bias=-3.0 (bf16 0xC040 = [0x40, 0xC0])
    // code=5: w = 2.0 * 5 + (-3.0) = 7.0
    let params = AffineParams {
        bits: 4,
        group_size: 32,
        storage: CodeStorage::U32Le,
        rows: 1,
        cols: 32,
    };
    let codes = vec![5u32; 32];
    let packed = pack_codes_for_test(&codes, &params);

    // 2.0f32: bits = 0x4000_0000, bf16 = 0x4000 = [0x00, 0x40]
    let scale_le = f32_to_bf16_le(2.0);
    // -3.0f32: bits = 0xC040_0000, bf16 = 0xC040 = [0x40, 0xC0]
    let bias_le = f32_to_bf16_le(-3.0);

    // Verify our bf16_to_f32 decodes these correctly.
    assert_eq!(bf16_to_f32(scale_le), 2.0_f32);
    assert_eq!(bf16_to_f32(bias_le), -3.0_f32);

    let scales: Vec<u8> = scale_le.to_vec();
    let biases: Vec<u8> = bias_le.to_vec();

    let out = dequant_vec(&params, &packed, &scales, &biases).unwrap();
    for &v in &out {
        assert!((v - 7.0_f32).abs() < 1e-4, "expected 7.0, got {v}");
    }
}
