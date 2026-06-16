use super::*;
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};
use crate::turboquant::{turbo_dequantize, turbo_quantize_v};

/// `original_shape: [i32; 4]` is 16 B inline — no heap allocation per PlanarBlocks.
///
/// Before perf(types): `original_shape: Vec<i32>` was 24 B stack + 16 B heap alloc.
/// After: 16 B inline in the struct.
#[test]
fn planar_blocks_original_shape_is_inline_array() {
    assert_eq!(
        size_of::<[i32; 4]>(),
        16,
        "original_shape must be 16 bytes (4 × i32)"
    );
    let shape = [1i32, 1, 1, 32];
    let data = vec![0.1_f32; 32];
    let blocks = planar_quantize(&data, GROUP_SIZE, 4, &shape).unwrap();
    assert_eq!(blocks.original_shape, shape);
}

// Gaussian-ish data via Box-Muller using LCG.
fn gaussian_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut lcg_u = || -> f32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to (0, 1) exclusive.
        ((state >> 33) as f32 + 0.5) / (u32::MAX as f32 + 1.0)
    };

    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = lcg_u();
        let u2 = lcg_u();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
        out.push(z0);
        if out.len() < n {
            out.push(z1);
        }
    }
    out
}

/// Roundtrip on [1, 4, 128, 64] f32 in [-1.0, 1.0]: max abs error < 0.10.
///
/// PlanarQuant uses per-pair scales which are finer-grained than TurboQuant's
/// per-block scales, so this tolerance (0.10) is tighter than TurboQuant's (0.15).
#[test]
fn planar_quantize_then_dequantize_within_tolerance() {
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);

    let blocks = planar_quantize(&data, GROUP_SIZE, 4, &shape).expect("planar_quantize failed");
    let recon = planar_dequantize(&blocks).expect("planar_dequantize failed");

    assert_eq!(recon.len(), n);

    let max_err = data
        .iter()
        .zip(recon.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.10,
        "PlanarQuant roundtrip max abs error {max_err:.6} exceeds 0.10"
    );
}

/// Codebook size = 16 and each R_k^T R_k ≈ I within 1e-5.
#[test]
fn planar_rotation_codebook_has_16_entries_orthogonal() {
    let cb = planar_rotation_codebook();
    assert_eq!(cb.len(), 16, "codebook must have 16 entries");

    for (k, entry) in cb.iter().enumerate() {
        let [c, neg_s, s, c2] = *entry;
        // R = [[c, -s], [s, c]]; R^T R = I
        let rtr_00 = c.mul_add(c, s * s);
        let rtr_11 = neg_s.mul_add(neg_s, c2 * c2);
        let rtr_01 = c.mul_add(neg_s, s * c2);
        let rtr_10 = s.mul_add(c, c2 * neg_s);

        let tol = 1e-5_f32;
        assert!(
            (rtr_00 - 1.0).abs() < tol,
            "codebook[{k}] R^T R [0,0] = {rtr_00} (expected 1)"
        );
        assert!(
            (rtr_11 - 1.0).abs() < tol,
            "codebook[{k}] R^T R [1,1] = {rtr_11} (expected 1)"
        );
        assert!(
            rtr_01.abs() < tol,
            "codebook[{k}] R^T R [0,1] = {rtr_01} (expected 0)"
        );
        assert!(
            rtr_10.abs() < tol,
            "codebook[{k}] R^T R [1,0] = {rtr_10} (expected 0)"
        );
    }
}

/// PlanarQuant 4-bit max abs error < TurboQuant V4 max abs error on Gaussian input.
///
/// PlanarQuant uses per-pair scales (finer grain than TurboQuant's per-block)
/// and chooses the rotation that minimizes per-pair reconstruction error. This
/// guarantees strictly lower or equal max error vs TurboQuant on any input.
///
/// If this test fails, the rotation selection or scale encoding is broken.
#[test]
fn planar_better_than_turbo_v4_on_gaussian() {
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = gaussian_data(n, 0xCAFE_BABE_u64);

    // PlanarQuant 4-bit.
    let planar_blocks =
        planar_quantize(&data, GROUP_SIZE, 4, &shape).expect("planar_quantize failed");
    let planar_recon = planar_dequantize(&planar_blocks).expect("planar_dequantize failed");
    let planar_max_err = data
        .iter()
        .zip(planar_recon.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    // TurboQuant V4 (no rotation, per-block scale).
    let turbo_blocks = turbo_quantize_v(&data, 4, &shape).expect("turbo_quantize_v failed");
    let turbo_recon = turbo_dequantize(&turbo_blocks).expect("turbo_dequantize failed");
    let turbo_max_err = data
        .iter()
        .zip(turbo_recon.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        planar_max_err < turbo_max_err,
        "PlanarQuant max err {planar_max_err:.6} >= TurboQuant max err {turbo_max_err:.6} — \
         rotation selection or per-pair scale bug: PlanarQuant must do better than per-block TurboQuant"
    );
}

// ── Cosine-similarity gate ────────────────────────────────────────────────────

// Fixture shape [1, 4, 128, 64] = 32 768 elements; rows are head_dim=64 slices.
const COSINE_SHAPE: [i32; 4] = [1, 4, 128, 64];
const COSINE_HEAD_DIM: usize = 64;

fn cosine_fixture() -> Vec<f32> {
    let n: usize = COSINE_SHAPE.iter().map(|&d| d as usize).product();
    lcg_data(n, TEST_SEED)
}

/// PlanarQuant V4 (Planar V-side) cosine gate: mean cosine ≥ 0.9942.
///
/// Threshold: per ../multi-turboquant README matrix row `planar4` = 0.9952 − 0.001.
#[test]
fn planar_v4_cosine_gate() {
    let data = cosine_fixture();
    let blocks =
        planar_quantize(&data, GROUP_SIZE, 4, &COSINE_SHAPE).expect("planar_quantize failed");
    let decoded = planar_dequantize(&blocks).expect("planar_dequantize failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9942,
        // per ../multi-turboquant README matrix row `planar4` = 0.9952 − 0.001
        "PlanarQuant V4 (Planar) mean cosine {:.6} < 0.9942",
        stats.mean,
    );
}

// ── K-axis PlanarQuant cosine + parity gates ──────────────────────────────────
//
// Step 0 kernel-share decision: PlanarQuant is axis-agnostic at the kernel
// input level (flat `[B, kv_h, S, D]`, `D % 32 == 0`), so the K-side codec
// reuses the same scalar `planar_quantize` / `planar_dequantize` functions
// AND the same MSL kernel binary. The cosine and parity expectations for the
// K side are therefore identical to the V row.

/// K-axis PlanarQuant cosine gate: mean cosine ≥ 0.9942.
#[test]
fn planar_k_cosine_gate() {
    let data = cosine_fixture();
    let blocks =
        planar_quantize(&data, GROUP_SIZE, 4, &COSINE_SHAPE).expect("planar_quantize K failed");
    let decoded = planar_dequantize(&blocks).expect("planar_dequantize K failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9942,
        "PlanarQuant K-axis mean cosine {:.6} < 0.9942 (shared kernel with V row)",
        stats.mean,
    );
}

/// K-axis vectorized parity gate.
#[test]
fn planar_k_vectorized_parity() {
    use crate::test_utils::vectorized_parity_check;
    let data = cosine_fixture();
    let shape = COSINE_SHAPE;
    vectorized_parity_check(
        |inp| {
            let blocks = planar_quantize(inp, GROUP_SIZE, 4, &shape).expect("cpu encode");
            planar_dequantize(&blocks).expect("cpu decode")
        },
        |inp| {
            // K-side via same axis-agnostic scalar codec — Step 0 decision.
            let blocks = planar_quantize(inp, GROUP_SIZE, 4, &shape).expect("k-side encode");
            planar_dequantize(&blocks).expect("k-side decode")
        },
        &data,
        1e-6,
        "planar_k_axis",
    );
}

// ── Codebook bit-exactness gates ──────────────────────────────────────────────

/// PlanarQuant V4 rotation codebook: bit patterns must match MSL-embedded constants.
///
/// `build_msl_header()` in `planarquant_msl.rs` embeds the rotation codebook by
/// calling `planar_rotation_codebook()` and formatting each f32 as its exact bit
/// pattern. If the CPU function drifts from these patterns, the GPU kernel silently
/// diverges from the CPU path.
///
/// Expected bit patterns verified via `f32::to_bits()` on `planar_rotation_codebook()`
/// output (Rust f32 `cos`/`sin` of `k * PI / 16` for k in 0..15).
#[test]
fn cb4_rotation_codebook_bit_exact() {
    let cb = planar_rotation_codebook();
    assert_eq!(cb.len(), 16, "rotation codebook must have 16 entries");

    // Each entry: [cos(theta), -sin(theta), sin(theta), cos(theta)]
    // theta_k = k * PI / 16, k = 0..15.
    let expected: [[u32; 4]; 16] = [
        [0x3F80_0000, 0x8000_0000, 0x0000_0000, 0x3F80_0000], // k=0
        [0x3F7B_14BE, 0xBE47_C5C2, 0x3E47_C5C2, 0x3F7B_14BE], // k=1
        [0x3F6C_835E, 0xBEC3_EF16, 0x3EC3_EF16, 0x3F6C_835E], // k=2
        [0x3F54_DB31, 0xBF0E_39DA, 0x3F0E_39DA, 0x3F54_DB31], // k=3
        [0x3F35_04F3, 0xBF35_04F3, 0x3F35_04F3, 0x3F35_04F3], // k=4
        [0x3F0E_39D9, 0xBF54_DB32, 0x3F54_DB32, 0x3F0E_39D9], // k=5
        [0x3EC3_EF15, 0xBF6C_835E, 0x3F6C_835E, 0x3EC3_EF15], // k=6
        [0x3E47_C5BC, 0xBF7B_14BF, 0x3F7B_14BF, 0x3E47_C5BC], // k=7
        [0xB33B_BD2E, 0xBF80_0000, 0x3F80_0000, 0xB33B_BD2E], // k=8
        [0xBE47_C5C2, 0xBF7B_14BE, 0x3F7B_14BE, 0xBE47_C5C2], // k=9
        [0xBEC3_EF18, 0xBF6C_835E, 0x3F6C_835E, 0xBEC3_EF18], // k=10
        [0xBF0E_39DC, 0xBF54_DB30, 0x3F54_DB30, 0xBF0E_39DC], // k=11
        [0xBF35_04F3, 0xBF35_04F3, 0x3F35_04F3, 0xBF35_04F3], // k=12
        [0xBF54_DB32, 0xBF0E_39D9, 0x3F0E_39D9, 0xBF54_DB32], // k=13
        [0xBF6C_8360, 0xBEC3_EF10, 0x3EC3_EF10, 0xBF6C_8360], // k=14
        [0xBF7B_14BF, 0xBE47_C5C1, 0x3E47_C5C1, 0xBF7B_14BF], // k=15
    ];

    for (k, (entry, exp)) in cb.iter().zip(expected.iter()).enumerate() {
        for (j, (&v, &exp_bits)) in entry.iter().zip(exp.iter()).enumerate() {
            assert_eq!(
                v.to_bits(),
                exp_bits,
                "ROT_CB[{k}][{j}] bit pattern: got 0x{:08X} expected 0x{exp_bits:08X}",
                v.to_bits(),
            );
        }
    }
}

// ── Planar3 (3-bit) CPU codec tests ───────────────────────────────────────────

/// Planar3 3-bit roundtrip within tolerance.
///
/// Per-pair rotation + 3-bit Lloyd-Max codebook. Tolerance tighter than raw
/// 3-bit TurboQuant because per-pair scales and rotation reduce error.
#[test]
fn planar_v3_cpu_roundtrip_within_tolerance() {
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);

    let blocks = planar_quantize(&data, GROUP_SIZE, 3, &shape).expect("planar_quantize v3 failed");
    let recon = planar_dequantize(&blocks).expect("planar_dequantize v3 failed");

    assert_eq!(recon.len(), n);

    let max_err = data
        .iter()
        .zip(recon.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    // 3-bit has coarser quantization than 4-bit; 0.20 is appropriate.
    assert!(
        max_err < 0.20,
        "Planar3 roundtrip max abs error {max_err:.6} exceeds 0.20"
    );
}

/// Planar3 cosine gate.
///
/// Threshold: measured mean cosine 0.999956 on LCG fixture at TEST_SEED; threshold = 0.9989.
/// (High cosine is expected: per-pair rotation + per-pair scale compresses small pairs
/// very well, even at 3 bits.)
#[test]
fn planar_v3_cosine_gate() {
    let data = cosine_fixture();
    let blocks =
        planar_quantize(&data, GROUP_SIZE, 3, &COSINE_SHAPE).expect("planar_quantize v3 failed");
    let decoded = planar_dequantize(&blocks).expect("planar_dequantize v3 failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9989,
        // measured mean = 0.999956 − 0.001 = 0.998956 → rounded down to 0.9989
        "Planar3 mean cosine {:.6} < 0.9989",
        stats.mean,
    );
}

/// Planar3 pack/unpack parity.
///
/// CPU encode then CPU decode must reconstruct exactly the same output
/// when run twice on the same input (deterministic codec path).
#[test]
fn planar_v3_cpu_parity_deterministic() {
    use crate::test_utils::vectorized_parity_check;
    let data = cosine_fixture();
    let shape = COSINE_SHAPE;
    vectorized_parity_check(
        |inp| {
            let blocks = planar_quantize(inp, GROUP_SIZE, 3, &shape).expect("v3 encode pass 1");
            planar_dequantize(&blocks).expect("v3 decode pass 1")
        },
        |inp| {
            let blocks = planar_quantize(inp, GROUP_SIZE, 3, &shape).expect("v3 encode pass 2");
            planar_dequantize(&blocks).expect("v3 decode pass 2")
        },
        &data,
        1e-6,
        "planar_v3_cpu_parity",
    );
}

// ── Cross-path packing parity (CPU bytes ↔ GPU word convention) ───────────────
//
// The GPU MSL kernels (`planarquant_msl.rs`) read the `codes` buffer as a flat
// `[n_groups * 4]` array of u32 words, extracting element `e` of a group via the
// shared "Planar3 pack convention":
//
//   bits=4: word = group*4 + e/8,  shift = (e%8)*4,  mask = 0xF  (8 vals/u32)
//   bits=3: word = group*4 + e/10, shift = (e%10)*3, mask = 0x7  (10 vals/u32)
//
// `QuantPlanarV` (storage/quant_planar_v.rs) hydrates a CPU-encoded layer onto
// the GPU by reinterpreting the CPU `PlanarBlocks::codes` byte vector as those
// u32 words. For that round-trip to be lossless the CPU encoder must emit codes
// in the SAME word convention the GPU reads. These tests prove byte-stream
// path-independence at the index level.

/// Re-extract one element's index from CPU-encoded `codes` bytes using the GPU
/// MSL word convention (`vals_per_word = 32 / bits`, group = 4 u32 words).
fn gpu_word_extract_index(codes: &[u8], group: usize, elem: usize, bits: u8) -> u8 {
    // CPU writes `code_bytes_per_block` bytes per group; GPU reads 4 u32/group.
    // Reinterpret the group's bytes as little-endian u32 words (the same view
    // `Array::from_bytes(.., Dtype::U32)` yields on Apple Silicon).
    // Intentionally hand-rolled, independent of `unpack_index`, so the parity
    // check in callers is non-circular (calling `unpack_index` here would make
    // the test vacuous).
    let vals_per_word = 32 / bits as usize;
    let word_in_group = elem / vals_per_word;
    let shift = (elem % vals_per_word) * bits as usize;
    let mask = (1u32 << bits) - 1;
    // 4 u32 words per group regardless of bits (3-bit and 4-bit both 4 words).
    let byte_base = group * 16 + word_in_group * 4;
    let mut word = 0u32;
    for (i, b) in codes.iter().skip(byte_base).take(4).copied().enumerate() {
        word |= u32::from(b) << (i * 8);
    }
    ((word >> shift) & mask) as u8
}

/// Read one element index out of CPU-encoded `codes` via the codec's own
/// `unpack_index` — the encoder's exact inverse.
fn cpu_extract_index(codes: &[u8], group: usize, elem: usize, bits: u8) -> u8 {
    // 4 u32 words = 16 bytes per block, for both 3-bit and 4-bit.
    let block = &codes[group * 16..(group + 1) * 16];
    unpack_index(block, elem, bits)
}

/// planar4 control: CPU dense byte packing and the GPU 8-vals/u32 word
/// convention extract identical indices for every element. 4-bit must stay
/// byte-identical across the CPU/GPU boundary — this guards the fix from
/// regressing the (already-correct) 4-bit path.
#[test]
fn planar4_cpu_bytes_match_gpu_word_convention() {
    let shape = [1i32, 1, 1, 32]; // exactly one group
    let data = lcg_data(GROUP_SIZE, 0x0102_0304_u64);
    let blocks = planar_quantize(&data, GROUP_SIZE, 4, &shape).expect("planar4 encode");
    assert_eq!(
        blocks.codes.len(),
        16,
        "4-bit: 32 elems × 4 bits = 16 bytes"
    );

    for elem in 0..GROUP_SIZE {
        let cpu = cpu_extract_index(&blocks.codes, 0, elem, 4);
        let gpu = gpu_word_extract_index(&blocks.codes, 0, elem, 4);
        assert_eq!(
            cpu, gpu,
            "planar4 elem {elem}: CPU index {cpu} != GPU word-convention index {gpu} — \
             4-bit packing must be path-independent"
        );
    }
}

/// planar3 cross-path parity: CPU-encoded `codes` bytes must yield the SAME
/// element indices whether read with the CPU dense unpacker or the GPU
/// 10-vals/u32 word convention. Without a unified packing scheme an SSD
/// spill (CPU encode) → hydrate (GPU read) silently corrupts the V cache.
///
/// FAILS on dense CPU packing (12 bytes/group, `bit_offset = elem * 3`);
/// PASSES once the CPU encoder emits the 10-vals/u32 word convention.
#[test]
fn planar3_cpu_bytes_match_gpu_word_convention() {
    let shape = [1i32, 1, 1, 32]; // exactly one group
    let data = lcg_data(GROUP_SIZE, 0x0a0b_0c0d_u64);
    let blocks = planar_quantize(&data, GROUP_SIZE, 3, &shape).expect("planar3 encode");

    let mut mismatches = 0usize;
    for elem in 0..GROUP_SIZE {
        let cpu = cpu_extract_index(&blocks.codes, 0, elem, 3);
        let gpu = gpu_word_extract_index(&blocks.codes, 0, elem, 3);
        if cpu != gpu {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "planar3: {mismatches}/{GROUP_SIZE} element indices differ between the CPU unpacker \
         and the GPU 10-vals/u32 word convention — the CPU/GPU byte streams are not \
         path-independent, so an SSD spill/hydrate across the boundary corrupts V codes"
    );
}

/// planar3 full byte-stream round-trip across the CPU/GPU boundary.
///
/// Encode on CPU → reinterpret the code bytes via the GPU word convention →
/// dequantize in rotated space with the same scales/rotations the CPU stored.
/// The reconstruction error must be quantization-noise small, not the ~1.x
/// cross-decode error seen when the two packings disagree.
#[test]
fn planar3_cross_path_decode_is_quant_noise() {
    let shape = [1i32, 4, 32, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xBADD_F00D_u64);

    let blocks = planar_quantize(&data, GROUP_SIZE, 3, &shape).expect("planar3 encode");

    // Reference: the codec's own CPU decode (the correctness anchor).
    let cpu_recon = planar_dequantize(&blocks).expect("planar3 cpu decode");

    // Cross-path: decode the SAME stored codes/scales/rotations but pull each
    // code index via the GPU 10-vals/u32 word convention instead of the CPU
    // dense unpacker. If the packings agree, this matches cpu_recon exactly.
    let rot_cb = planar_rotation_codebook();
    let codebook = lloyd_gaussian_codebook(3).expect("3-bit codebook");
    let n_blocks = n / GROUP_SIZE;
    let pairs_per_block = GROUP_SIZE / 2;
    let rot_bytes_per_block = pairs_per_block / 2;
    let mut cross_recon = vec![0.0_f32; n];

    for block in 0..n_blocks {
        for pair in 0..pairs_per_block {
            let scale = blocks.scales[block * pairs_per_block + pair];
            let rot_byte = blocks.rotations[block * rot_bytes_per_block + pair / 2];
            let rot_idx = ((rot_byte >> ((pair % 2) * 4)) & 0xF) as usize;
            let entry = &rot_cb[rot_idx];

            let elem_a = pair * 2;
            let elem_b = pair * 2 + 1;
            let idx_a = gpu_word_extract_index(&blocks.codes, block, elem_a, 3) as usize;
            let idx_b = gpu_word_extract_index(&blocks.codes, block, elem_b, 3) as usize;
            let ya = codebook[idx_a] * scale;
            let yb = codebook[idx_b] * scale;
            // R^T = [[c, s], [-s, c]] (entry = [c, -s, s, c]).
            let a = entry[0] * ya + entry[2] * yb;
            let b = entry[1] * ya + entry[3] * yb;
            cross_recon[block * GROUP_SIZE + elem_a] = a;
            cross_recon[block * GROUP_SIZE + elem_b] = b;
        }
    }

    let cross_err = cpu_recon
        .iter()
        .zip(cross_recon.iter())
        .map(|(&c, &x)| (c - x).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        cross_err < 1e-6,
        "planar3 cross-path (CPU encode → GPU-convention decode) max abs error {cross_err:.6} \
         vs CPU decode — packing is not path-independent; an SSD hydrate would corrupt V"
    );
}
