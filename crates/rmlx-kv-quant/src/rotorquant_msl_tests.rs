//! rotor3 / rotor4 MSL kernel parity tests.
//!
//! All GPU tests are `#[ignore]`-gated. They run only when explicitly invoked:
//!
//! ```text
//! cargo test -p rmlx-kv-quant --lib -- --ignored rotorquant_msl --test-threads=1
//! ```
//!
//! `RMLX_SKIP_GPU=1` causes GPU tests to exit silently even with
//! `--include-ignored`.

use super::*;
use crate::clifford::make_rotor_table;
use crate::rotorquant::{
    rotor3_decode, rotor3_encode, rotor3_k_decode, rotor3_k_encode, rotor4_decode, rotor4_encode,
    rotor4_k_decode, rotor4_k_encode,
};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};

/// Build a test `Array` from a f32 slice and shape.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in test scaffolding"
)]
#[allow(
    unsafe_code,
    reason = "Metal FFI test helper: reinterpret f32 slice as bytes for Array::from_bytes; \
              slice lifetime is tied to `data`, no aliasing — safe by construction"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: `f32` has no padding, alignment is 4; reinterpreting as `u8` is
    // well-defined. The byte slice borrows from `data` for the duration of
    // this call. `Array::from_bytes` copies the bytes immediately.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

/// Materialise an `Array` to `Vec<f32>` via the MLX graph executor.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in test scaffolding"
)]
#[allow(
    clippy::unwrap_used,
    reason = "chunks_exact(4) + try_into are infallible given the f32 element size invariant"
)]
fn array_to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("array materialise");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// ── rotor3 V parity ───────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_v3_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    // head_dim=129 is intentionally NOT a multiple of 3 to exercise tail-pad.
    // n_tokens=32, n_groups=ceil(129/3)=43.
    let head_dim: usize = 129;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x1810_BEEF_u64);

    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    vectorized_parity_check(
        |input| {
            let (codes, scales, norms) =
                rotor3_encode(input, &rotors, head_dim).expect("rotor3 cpu encode");
            rotor3_decode(&codes, &scales, &norms, &rotors, head_dim).expect("rotor3 cpu decode")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, norms) =
                rotor_quantize_v3_gpu(&arr, &rotors_arr, head_dim, Device::Gpu)
                    .expect("rotor3 gpu encode");
            let out = rotor_dequantize_v3_gpu(
                &codes,
                &scales,
                &norms,
                &rotors_arr,
                head_dim,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("rotor3 gpu decode");
            array_to_f32_vec(&out)
        },
        &data,
        5e-3_f32,
        "rotor3 CPU vs MSL",
    );
}

// ── rotor4 V parity ───────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_v4_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 129;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x1811_BEEF_u64);

    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    vectorized_parity_check(
        |input| {
            let (codes, scales, norms) =
                rotor4_encode(input, &rotors, head_dim).expect("rotor4 cpu encode");
            rotor4_decode(&codes, &scales, &norms, &rotors, head_dim).expect("rotor4 cpu decode")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, norms) =
                rotor_quantize_v4_gpu(&arr, &rotors_arr, head_dim, Device::Gpu)
                    .expect("rotor4 gpu encode");
            let out = rotor_dequantize_v4_gpu(
                &codes,
                &scales,
                &norms,
                &rotors_arr,
                head_dim,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("rotor4 gpu decode");
            array_to_f32_vec(&out)
        },
        &data,
        5e-3_f32,
        "rotor4 CPU vs MSL",
    );
}

// ── rotor3 K parity (no QJL — QJL is CPU-only) ────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_k3_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    // K-side uses the same rotor codec as V-side; QJL is opt-in (None here).
    let head_dim: usize = 128;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x1812_BEEF_u64);

    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    vectorized_parity_check(
        |input| {
            // K-side encode with qjl_s_matrix = None → identical to rotor3_encode
            // except for the empty qjl side-channels that the decode also drops.
            let (codes, scales, norms, qjl_packed, qjl_norms) =
                rotor3_k_encode(input, &rotors, head_dim, None).expect("rotor3 k cpu encode");
            rotor3_k_decode(
                &codes,
                &scales,
                &norms,
                &rotors,
                head_dim,
                &qjl_packed,
                &qjl_norms,
                None,
            )
            .expect("rotor3 k cpu decode")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, norms) =
                rotor_quantize_v3_gpu(&arr, &rotors_arr, head_dim, Device::Gpu)
                    .expect("rotor3 gpu encode");
            let out = rotor_dequantize_v3_gpu(
                &codes,
                &scales,
                &norms,
                &rotors_arr,
                head_dim,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("rotor3 gpu decode");
            array_to_f32_vec(&out)
        },
        &data,
        5e-3_f32,
        "rotor3-K (QJL off) CPU vs MSL",
    );
}

// ── rotor4 K parity (no QJL) ──────────────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_k4_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 128;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x1813_BEEF_u64);

    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    vectorized_parity_check(
        |input| {
            let (codes, scales, norms, qjl_packed, qjl_norms) =
                rotor4_k_encode(input, &rotors, head_dim, None).expect("rotor4 k cpu encode");
            rotor4_k_decode(
                &codes,
                &scales,
                &norms,
                &rotors,
                head_dim,
                &qjl_packed,
                &qjl_norms,
                None,
            )
            .expect("rotor4 k cpu decode")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, norms) =
                rotor_quantize_v4_gpu(&arr, &rotors_arr, head_dim, Device::Gpu)
                    .expect("rotor4 gpu encode");
            let out = rotor_dequantize_v4_gpu(
                &codes,
                &scales,
                &norms,
                &rotors_arr,
                head_dim,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("rotor4 gpu decode");
            array_to_f32_vec(&out)
        },
        &data,
        5e-3_f32,
        "rotor4-K (QJL off) CPU vs MSL",
    );
}

// ── Cross-encoder bit-equivalence parity ─────────────────────────────────────
//
// The four tests above run `cpu_encode → cpu_decode` vs
// `gpu_encode → gpu_decode`. A coordinated sign error in the forward
// rotation `M(R)` and its transpose `M(R)^T` would cancel in the round-trip
// while corrupting the production hot path `gpu_encode → CPU vs.dequant()`.
//
// The tests below pin the GPU encoder directly against the CPU encoder:
//
//   1. `..._gpu_encode_matches_cpu_encode` — same input fed through
//      both encoders, assert ≥ 95% code agreement.  Both sides take their
//      scale from the three stored grade-1 components, so what is left is
//      the assignment rule: the CPU scans for the nearest centroid, the GPU
//      compares against the midpoint boundaries, and the two can disagree
//      only at a boundary — every disagreement is ±1 from the CPU centroid
//      index, which `assert_one_step_slips` pins.  Scales + norms must still
//      agree within (1e-5 abs + 1e-4 rel) per element.  A sign error in
//      `M(R)` would manifest as ≥30% disagreement, far below the gate.
//   2. `..._gpu_encode_then_cpu_decode_round_trip` — gpu encode + CPU
//      decode, compare against the CPU encode + CPU decode of the same
//      input within 5e-3 (the same tolerance the original 4 round-trip
//      tests use for GPU↔GPU comparison).  Comparing against the original
//      input would just measure 3-bit quantization noise (~0.1–0.3
//      max-abs); the load-bearing assertion is that swapping the decoder
//      side from GPU to CPU does not introduce systematic drift.

#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn gpu_encode_v3(
    input: &[f32],
    rotors_arr: &Array,
    head_dim: usize,
    shape: &[i32],
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let arr = make_f32_array(input, shape);
    let (codes_arr, scales_arr, norms_arr) =
        rotor_quantize_v3_gpu(&arr, rotors_arr, head_dim, Device::Gpu).expect("rotor3 gpu encode");
    #[allow(
        clippy::cast_sign_loss,
        reason = "shape dims non-negative by construction"
    )]
    let n_tokens = shape[0] as usize;
    let n_groups = head_dim.div_ceil(ROTOR3_GROUP_SIZE);
    rotor_gpu_outputs_to_cpu(
        &codes_arr,
        &scales_arr,
        &norms_arr,
        n_tokens,
        n_groups,
        crate::rotorquant::row_words_for(head_dim, 3),
    )
    .expect("rotor3 gpu outputs to cpu")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn gpu_encode_v4(
    input: &[f32],
    rotors_arr: &Array,
    head_dim: usize,
    shape: &[i32],
) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let arr = make_f32_array(input, shape);
    let (codes_arr, scales_arr, norms_arr) =
        rotor_quantize_v4_gpu(&arr, rotors_arr, head_dim, Device::Gpu).expect("rotor4 gpu encode");
    #[allow(
        clippy::cast_sign_loss,
        reason = "shape dims non-negative by construction"
    )]
    let n_tokens = shape[0] as usize;
    let n_groups = head_dim.div_ceil(ROTOR3_GROUP_SIZE);
    rotor_gpu_outputs_to_cpu(
        &codes_arr,
        &scales_arr,
        &norms_arr,
        n_tokens,
        n_groups,
        crate::rotorquant::row_words_for(head_dim, 4),
    )
    .expect("rotor4 gpu outputs to cpu")
}

/// Every code of a dense-plane stream, row by row.
///
/// Reads through [`crate::code_plane::read_code`] rather than shifting the
/// words here: a second unpack rule in the test would agree with itself while
/// disagreeing with the store.
fn plane_codes(words: &[u32], head_dim: usize, bits: u8) -> Vec<u8> {
    let row_words = crate::rotorquant::row_words_for(head_dim, bits);
    let per_row = crate::rotorquant::codes_per_row(head_dim);
    let rows = words.len() / row_words;
    let mut out = Vec::with_capacity(rows * per_row);
    for r in 0..rows {
        let row = &words[r * row_words..(r + 1) * row_words];
        for i in 0..per_row {
            out.push(crate::code_plane::read_code(row, i, bits));
        }
    }
    out
}

/// Compare two dense code planes code by code. Returns (n_pairs, n_disagree).
fn count_code_disagreements(a: &[u32], b: &[u32], head_dim: usize, bits: u8) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "code length mismatch");
    let (ca, cb) = (
        plane_codes(a, head_dim, bits),
        plane_codes(b, head_dim, bits),
    );
    let n_disagree = ca.iter().zip(cb.iter()).filter(|(x, y)| x != y).count();
    (ca.len(), n_disagree)
}

/// Assert that every disagreeing code differs from its CPU counterpart by
/// exactly ±1 in the codebook index space. A sign error in `M(R)`, or a
/// packing-order bug, would produce arbitrary multi-step jumps; the two
/// assignment rules — the CPU's nearest-centroid scan against the GPU's
/// boundary comparisons — can only disagree at a boundary.
fn assert_one_step_slips(a: &[u32], b: &[u32], head_dim: usize, bits: u8, name: &str) {
    assert_eq!(a.len(), b.len(), "[{name}] code length mismatch");
    let (ca_all, cb_all) = (
        plane_codes(a, head_dim, bits),
        plane_codes(b, head_dim, bits),
    );
    let mut bad_jumps = 0usize;
    let mut total_diff = 0usize;
    for (&ca, &cb) in ca_all.iter().zip(cb_all.iter()) {
        let d = i32::from(cb) - i32::from(ca);
        if d != 0 {
            total_diff += 1;
            if d.abs() > 1 {
                bad_jumps += 1;
            }
        }
    }
    assert!(
        bad_jumps == 0,
        "[{name}] {bad_jumps} codes differ by >1 step ({total_diff} total diffs); \
         boundary-slip-only invariant violated — possible algorithmic divergence",
    );
}

/// Per-element scalar-pair tolerance check: `|a - b| <= abs_tol + rel_tol * |b|`.
fn assert_close(a: &[f32], b: &[f32], abs_tol: f32, rel_tol: f32, name: &str) {
    assert_eq!(a.len(), b.len(), "[{name}] length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let tol = rel_tol.mul_add(y.abs(), abs_tol);
        let err = (x - y).abs();
        assert!(
            err <= tol,
            "[{name}] mismatch at idx {i}: cpu={y:.6e} gpu={x:.6e} err={err:.2e} tol={tol:.2e}",
        );
    }
}

// ── rotor3 V: GPU encode vs CPU encode ────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_v3_gpu_encode_matches_cpu_encode() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 129;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x181A_BEEF_u64);
    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    let (cpu_codes, cpu_scales, cpu_norms) =
        rotor3_encode(&data, &rotors, head_dim).expect("rotor3 cpu encode");
    let (gpu_codes, gpu_scales, gpu_norms_per_grp) =
        gpu_encode_v3(&data, &rotors_arr, head_dim, &shape);

    // Codes: >=95% agreement. Both sides now take their scale from the three
    // stored grade-1 components, so what is left is the assignment rule — the
    // CPU's nearest-centroid scan against the GPU's boundary comparisons — which
    // can only disagree at a boundary.
    let (n_pairs, n_disagree) = count_code_disagreements(&cpu_codes, &gpu_codes, head_dim, 3);
    let agree_pct = 100.0 * ((n_pairs - n_disagree) as f64) / (n_pairs as f64);
    assert!(
        agree_pct >= 95.0,
        "rotor3 V code agreement {agree_pct:.4}% < 95% (disagree {n_disagree} / {n_pairs}, boundary slack is allowed but a sign bug would push this well below 70%)",
    );
    assert_one_step_slips(&cpu_codes, &gpu_codes, head_dim, 3, "rotor3 V");

    // Scales: per-element within (1e-5 abs + 1e-4 rel).
    assert_close(&gpu_scales, &cpu_scales, 1e-5, 1e-4, "rotor3 V scales");

    // GPU norms_per_group are per-(token, group); CPU norms are per-token.
    // Per [`rotor_gpu_outputs_to_cpu`] the GPU vector returned is already
    // deduplicated to per-token (first slot of each token).
    assert_close(&gpu_norms_per_grp, &cpu_norms, 1e-5, 1e-4, "rotor3 V norms");
}

// ── rotor4 V: GPU encode vs CPU encode ────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_v4_gpu_encode_matches_cpu_encode() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 129;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x181B_BEEF_u64);
    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    let (cpu_codes, cpu_scales, cpu_norms) =
        rotor4_encode(&data, &rotors, head_dim).expect("rotor4 cpu encode");
    let (gpu_codes, gpu_scales, gpu_norms_per_grp) =
        gpu_encode_v4(&data, &rotors_arr, head_dim, &shape);

    let (n_pairs, n_disagree) = count_code_disagreements(&cpu_codes, &gpu_codes, head_dim, 4);
    let agree_pct = 100.0 * ((n_pairs - n_disagree) as f64) / (n_pairs as f64);
    assert!(
        agree_pct >= 95.0,
        "rotor4 V code agreement {agree_pct:.4}% < 95% (disagree {n_disagree} / {n_pairs}, boundary slack is allowed but a sign bug would push this well below 70%)",
    );
    assert_one_step_slips(&cpu_codes, &gpu_codes, head_dim, 4, "rotor4 V");
    assert_close(&gpu_scales, &cpu_scales, 1e-5, 1e-4, "rotor4 V scales");
    assert_close(&gpu_norms_per_grp, &cpu_norms, 1e-5, 1e-4, "rotor4 V norms");
}

// ── rotor3 K (QJL off): GPU encode vs CPU encode ──────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_k3_gpu_encode_matches_cpu_encode() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 128;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x181C_BEEF_u64);
    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    // K-side encode with qjl_s_matrix = None → identical to rotor3_encode in
    // (codes, scales, norms); we ignore qjl_packed / qjl_norms.
    let (cpu_codes, cpu_scales, cpu_norms, _qjl_packed, _qjl_norms) =
        rotor3_k_encode(&data, &rotors, head_dim, None).expect("rotor3 k cpu encode");
    let (gpu_codes, gpu_scales, gpu_norms_per_grp) =
        gpu_encode_v3(&data, &rotors_arr, head_dim, &shape);

    let (n_pairs, n_disagree) = count_code_disagreements(&cpu_codes, &gpu_codes, head_dim, 3);
    let agree_pct = 100.0 * ((n_pairs - n_disagree) as f64) / (n_pairs as f64);
    assert!(
        agree_pct >= 95.0,
        "rotor3 K code agreement {agree_pct:.4}% < 95% (disagree {n_disagree} / {n_pairs})",
    );
    assert_one_step_slips(&cpu_codes, &gpu_codes, head_dim, 3, "rotor3 K");
    assert_close(&gpu_scales, &cpu_scales, 1e-5, 1e-4, "rotor3 K scales");
    assert_close(&gpu_norms_per_grp, &cpu_norms, 1e-5, 1e-4, "rotor3 K norms");
}

// ── rotor4 K (QJL off): GPU encode vs CPU encode ──────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_k4_gpu_encode_matches_cpu_encode() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 128;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x181D_BEEF_u64);
    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    let (cpu_codes, cpu_scales, cpu_norms, _qjl_packed, _qjl_norms) =
        rotor4_k_encode(&data, &rotors, head_dim, None).expect("rotor4 k cpu encode");
    let (gpu_codes, gpu_scales, gpu_norms_per_grp) =
        gpu_encode_v4(&data, &rotors_arr, head_dim, &shape);

    let (n_pairs, n_disagree) = count_code_disagreements(&cpu_codes, &gpu_codes, head_dim, 4);
    let agree_pct = 100.0 * ((n_pairs - n_disagree) as f64) / (n_pairs as f64);
    assert!(
        agree_pct >= 95.0,
        "rotor4 K code agreement {agree_pct:.4}% < 95% (disagree {n_disagree} / {n_pairs})",
    );
    assert_one_step_slips(&cpu_codes, &gpu_codes, head_dim, 4, "rotor4 K");
    assert_close(&gpu_scales, &cpu_scales, 1e-5, 1e-4, "rotor4 K scales");
    assert_close(&gpu_norms_per_grp, &cpu_norms, 1e-5, 1e-4, "rotor4 K norms");
}

// ── Production-path round-trip: GPU encode + CPU decode ───────────────────────
//
// The hot path is `gpu_encode → vs.dequant()` (CPU). The four
// round-trip tests above do `gpu encode → gpu decode`. A coordinated sign
// error in the GPU encode and GPU decode would cancel, but the production
// path uses the CPU decoder and would expose the bug.

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_v3_gpu_encode_then_cpu_decode_round_trip() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 129;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x181E_BEEF_u64);
    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    // GPU encode + CPU decode: the production hot path.
    let (gpu_codes, gpu_scales, gpu_norms) = gpu_encode_v3(&data, &rotors_arr, head_dim, &shape);
    let mixed_dec = rotor3_decode(&gpu_codes, &gpu_scales, &gpu_norms, &rotors, head_dim)
        .expect("rotor3 cpu decode of gpu codes");

    // Reference: pure CPU enc + CPU dec.
    let (cpu_codes, cpu_scales, cpu_norms) =
        rotor3_encode(&data, &rotors, head_dim).expect("rotor3 cpu encode");
    let cpu_dec = rotor3_decode(&cpu_codes, &cpu_scales, &cpu_norms, &rotors, head_dim)
        .expect("rotor3 cpu decode");

    assert_eq!(mixed_dec.len(), cpu_dec.len(), "round-trip length mismatch");
    let max_err = mixed_dec
        .iter()
        .zip(cpu_dec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= 5e-3,
        "rotor3 V (GPU-encode → CPU-decode) vs (CPU-encode → CPU-decode) max-abs-error {max_err:.2e} > 5e-3 (production hot path swap)",
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: codec returns Ok for valid shapes used in scaffolding"
)]
fn rotor_v4_gpu_encode_then_cpu_decode_round_trip() {
    if skip_if_no_gpu_env() {
        return;
    }
    let head_dim: usize = 129;
    let n_tokens: usize = 32;
    let n_groups = head_dim.div_ceil(3);
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x181F_BEEF_u64);
    let rotors = make_rotor_table(0, 0, n_groups);
    let rotors_arr = make_f32_array(&rotors, &[(n_groups * 4) as i32]);
    let shape = [n_tokens as i32, head_dim as i32];

    let (gpu_codes, gpu_scales, gpu_norms) = gpu_encode_v4(&data, &rotors_arr, head_dim, &shape);
    let mixed_dec = rotor4_decode(&gpu_codes, &gpu_scales, &gpu_norms, &rotors, head_dim)
        .expect("rotor4 cpu decode of gpu codes");

    let (cpu_codes, cpu_scales, cpu_norms) =
        rotor4_encode(&data, &rotors, head_dim).expect("rotor4 cpu encode");
    let cpu_dec = rotor4_decode(&cpu_codes, &cpu_scales, &cpu_norms, &rotors, head_dim)
        .expect("rotor4 cpu decode");

    assert_eq!(mixed_dec.len(), cpu_dec.len(), "round-trip length mismatch");
    let max_err = mixed_dec
        .iter()
        .zip(cpu_dec.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err <= 5e-3,
        "rotor4 V (GPU-encode → CPU-decode) vs (CPU-encode → CPU-decode) max-abs-error {max_err:.2e} > 5e-3 (production hot path swap)",
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
fn hdr_probe_snapshots_match_builders() {
    assert_eq!(
        kernel_header_rotor3().expect("rotor3 header"),
        include_str!("metal/probes/rotorquant_rotor3.hdr.metal"),
        "stale snapshot: refresh metal/probes/rotorquant_rotor3.hdr.metal"
    );
    assert_eq!(
        kernel_header_rotor4().expect("rotor4 header"),
        include_str!("metal/probes/rotorquant_rotor4.hdr.metal"),
        "stale snapshot: refresh metal/probes/rotorquant_rotor4.hdr.metal"
    );
}
