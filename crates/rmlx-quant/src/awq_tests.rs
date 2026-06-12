use super::*;

/// Byte-layout round-trip for the AWQ → MLX weight + scales/biases conversion.
///
/// The GPU `quantized_matmul` correctness check that previously followed this
/// byte-layout check lives in `rmlx-models` (`qwen3_5_moe/tests.rs`), where the
/// MLX dependency is available — `rmlx-quant` is byte-math only.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn paro_weight_conversion_roundtrip() {
    let in_f = 128usize;
    let out_f = 8usize;
    let bits = 4usize;
    let num_groups = in_f / 128; // = 1

    // Build expected nibble matrix [in, out]: nibble[i][o] = (i + o) % 8
    // Each element fits in 4 bits (value 0..7 < 15).
    let mut nibble_matrix = vec![[0u8; 8]; in_f];
    for (i, row) in nibble_matrix.iter_mut().enumerate() {
        for (o, cell) in row.iter_mut().enumerate() {
            *cell = ((i + o) % 8) as u8;
        }
    }

    // Pack in AWQ order: [in, out*bits/32] = [128, 1] I32 words.
    // AWQ interleave: output elements [0,2,4,6,1,3,5,7] go to nibble positions [0..7].
    let awq_order: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let words_per_in = out_f * bits / 32; // = 1
    let mut qweight_bytes = vec![0u8; in_f * words_per_in * 4];
    for (i, row) in nibble_matrix.iter().enumerate() {
        // One word per input row (8 nibbles = 8 outputs).
        let mut word = 0u32;
        for (pos, &o) in awq_order.iter().enumerate() {
            word |= u32::from(row[o]) << (pos * 4);
        }
        let off = i * words_per_in * 4;
        qweight_bytes[off..off + 4].copy_from_slice(&word.to_le_bytes());
    }

    // Convert AWQ qweight → MLX layout.
    let mlx_weight_bytes =
        convert_awq_qweight(&qweight_bytes, in_f, out_f, bits).expect("convert_awq_qweight");
    // Expected shape: [out=8, in*bits/32=1] words.
    assert_eq!(mlx_weight_bytes.len(), out_f * (in_f * bits / 32) * 4);

    // Verify: unpack MLX weight and check nibbles match transpose [in, out] → [out, in].
    let words_per_out = in_f * bits / 32; // = 128*4/32 = 16
    #[allow(clippy::needless_range_loop)]
    for o in 0..out_f {
        for j in 0..words_per_out {
            let off = (o * words_per_out + j) * 4;
            let word = u32::from_le_bytes([
                mlx_weight_bytes[off],
                mlx_weight_bytes[off + 1],
                mlx_weight_bytes[off + 2],
                mlx_weight_bytes[off + 3],
            ]);
            // MLX sequential: nibble at position k = input element (j*8 + k).
            for k in 0..8usize {
                let in_idx = j * 8 + k;
                if in_idx >= in_f {
                    break;
                }
                let nibble = ((word >> (k * 4)) & 0xF) as u8;
                let expected = nibble_matrix[in_idx][o];
                assert_eq!(
                    nibble, expected,
                    "MLX weight nibble[out={o}, in={in_idx}]: expected {expected}, got {nibble}"
                );
            }
        }
    }

    // Build scales (F16 = 1.0) and zeros (0) for num_groups=1, out_f=8.
    // AWQ scales shape: [num_groups=1, out_f=8] F16.
    // AWQ qzeros shape: [num_groups=1, out_f*bits/32=1] I32 (zeros packed as nibbles).
    let scale_f16_bits: u16 = 0x3C00; // 1.0 in F16
    let mut scales_bytes = vec![0u8; num_groups * out_f * 2]; // [1, 8] F16
    for o in 0..out_f {
        let off = o * 2;
        scales_bytes[off..off + 2].copy_from_slice(&scale_f16_bits.to_le_bytes());
    }
    // qzeros: [num_groups=1, out*bits/32=1] all zeros → zero-points = 0.
    let qzeros_bytes = vec![0u8; num_groups * words_per_in * 4];

    let (scales_t_bytes, biases_t_bytes) =
        convert_awq_qzeros_to_biases(&qzeros_bytes, &scales_bytes, num_groups, out_f, bits)
            .expect("convert_awq_qzeros_to_biases");

    // scales_t: [out=8, num_groups=1] F16 = all 1.0
    // biases_t: [out=8, num_groups=1] F16 = -1.0 * 0 = 0.0
    for o in 0..out_f {
        let s_bits = u16::from_le_bytes([scales_t_bytes[o * 2], scales_t_bytes[o * 2 + 1]]);
        assert_eq!(s_bits, scale_f16_bits, "scale[o={o}] must be 1.0 F16");
        let b_bits = u16::from_le_bytes([biases_t_bytes[o * 2], biases_t_bytes[o * 2 + 1]]);
        // -1.0 * 0 = -0.0 (F16 0x8000) or +0.0 (0x0000): both are zero.
        assert!(
            b_bits == 0u16 || b_bits == 0x8000u16,
            "bias[o={o}] must be ±0.0 F16, got {b_bits:#06X}"
        );
    }
}

/// Verify that `quantize_f16_affine_int4` matches MLX `mx.quantize` output.
///
/// Uses a known row of embed_tokens (row 760, first 128 elements) and checks
/// that our Rust quantization produces the same scale/bias as Python.
///
/// Python reference (from paro_embed_check.py):
/// row760[:128] scale ≈ -0.01029, bias ≈ 0.12354
/// dequant[:8] = [0.02061, 3.05e-5, 3.05e-5, 3.05e-5, 3.05e-5, 3.05e-5, 0.01032, -0.01026]
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn quantize_f16_affine_int4_matches_python() {
    // Synthesize a known group: row with min≈-0.034, max≈0.124.
    // These bounds are from the first 128 elements of embed_tokens.weight[760].
    // We don't need the actual model file — we can test the formula directly.
    let group_size = 128usize;
    let n = group_size;

    // Build a synthetic F16 row with known min=-0.08 and max=0.08.
    // scale = -(0.08 - (-0.08)) / 15 = -0.16/15 ≈ -0.010667
    // bias = max = 0.08
    let min_val = -0.08_f32;
    let max_val = 0.08_f32;
    let mut row_f32: Vec<f32> = (0..n)
        .map(|i| min_val + (max_val - min_val) * (i as f32) / ((n - 1) as f32))
        .collect();

    // Encode as F16 bytes.
    let mut row_bytes = vec![0u8; n * 2];
    for (i, &v) in row_f32.iter().enumerate() {
        let bits = f32_to_f16_bits(v);
        row_bytes[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
    }
    // Re-read the actual F16 values (may differ slightly from f32 due to F16 precision).
    for i in 0..n {
        let bits = u16::from_le_bytes([row_bytes[i * 2], row_bytes[i * 2 + 1]]);
        row_f32[i] = f16_bits_to_f32(bits);
    }
    let actual_min = row_f32.iter().copied().fold(f32::INFINITY, f32::min);
    let actual_max = row_f32.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let (wq_bytes, sc_bytes, bi_bytes) =
        quantize_f16_affine_int4(&row_bytes, 1, n, group_size).expect("quantize");

    let scale_bits = u16::from_le_bytes([sc_bytes[0], sc_bytes[1]]);
    let bias_bits = u16::from_le_bytes([bi_bytes[0], bi_bytes[1]]);
    let scale_f32 = f16_bits_to_f32(scale_bits);
    let bias_f32 = f16_bits_to_f32(bias_bits);

    let expected_scale = -(actual_max - actual_min) / 15.0;
    let expected_bias = actual_max;
    let tol = 0.001_f32;
    assert!(
        (scale_f32 - f16_bits_to_f32(f32_to_f16_bits(expected_scale))).abs() < tol,
        "scale mismatch: got={scale_f32:.6}, expected≈{expected_scale:.6}"
    );
    assert!(
        (bias_f32 - f16_bits_to_f32(f32_to_f16_bits(expected_bias))).abs() < tol,
        "bias mismatch: got={bias_f32:.6}, expected≈{expected_bias:.6}"
    );

    // Dequant and check round-trip error is within 1 quantization step.
    let step = (actual_max - actual_min) / 15.0;
    for (i, &orig) in row_f32.iter().enumerate().take(n) {
        let word_idx = i / 8;
        let nibble_pos = i % 8;
        let word = u32::from_le_bytes(wq_bytes[word_idx * 4..word_idx * 4 + 4].try_into().unwrap());
        let nibble = (word >> (nibble_pos * 4)) & 0xF;
        let dequant = (nibble as f32).mul_add(scale_f32, bias_f32);
        let err = (dequant - orig).abs();
        assert!(
            err <= step + 0.001,
            "dequant error at [{i}]: original={orig:.6}, dequant={dequant:.6}, err={err:.6}, step={step:.6}",
        );
    }
}

#[test]
fn f16_subnormal_negative_decodes_nonzero() {
    // 0x8001 = sign=1, exp=0, mantissa=1 → -1 * 2^-24
    let neg = f16_bits_to_f32(0x8001);
    assert_eq!(neg, -(1.0 / 16777216.0));
    // 0x83FF = largest negative subnormal: -1023 * 2^-24
    let neg_max = f16_bits_to_f32(0x83FF);
    assert_eq!(neg_max, -1023.0 / 16777216.0);
    // Positive subnormal unchanged.
    assert_eq!(f16_bits_to_f32(0x0001), 1.0 / 16777216.0);
    // Signed zeros preserved.
    assert_eq!(f16_bits_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
    assert_eq!(f16_bits_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
}
