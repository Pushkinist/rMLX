//! Sequence-major layout round-trip tests for [`QuantKTurbo4`].
//!
//! The flat QuantKTurbo4 buffer accumulates one chunk per `append` at a
//! sequence offset (`prev_seq * words_per_seq`) while the dequant reshapes the
//! flat prefix as the logical head-major `[B, kv_h, S, D]`. The fix stores
//! every chunk sequence-major (reorder heads↔seq before quantizing, reorder
//! back on dequant). The round-trip tests below prove no head scramble on a
//! multi-append GQA cache; the GPU one is `#[ignore]` (Metal context) — run
//! with `--ignored --test-threads=1`.

use rmlx_mlx::{Array, Device, Dtype};

use super::QuantKTurbo4;
use crate::test_utils::skip_if_no_gpu_env;

fn new_turbo4(kv_h: i32, d: i32) -> QuantKTurbo4 {
    QuantKTurbo4 {
        blocks: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: vec![1, kv_h, 0, d],
        bits: 4,
        max_seq: 0,
    }
}

/// Distinct, small per-(head,token,dim) value so a head transposition (swaps in
/// a value differing by ≥ ~0.1) is obvious against q4 noise.
fn expected(h: i32, s: i32, d: i32) -> f32 {
    (h * 100 + s * 5 + d % 7) as f32 * 0.001
}

/// Head-major flat `[1, kv_h, seq, d]` chunk — the layout `append` receives.
fn head_major_chunk(kv_h: i32, seq: i32, d: i32, base_s: i32) -> Vec<f32> {
    let mut v = Vec::with_capacity((kv_h * seq * d) as usize);
    for h in 0..kv_h {
        for s in 0..seq {
            for dd in 0..d {
                v.push(expected(h, base_s + s, dd));
            }
        }
    }
    v
}

fn check(out: &[f32], kv_h: i32, s_total: i32, d: i32) -> f32 {
    let mut m = 0.0_f32;
    let mut i = 0usize;
    for h in 0..kv_h {
        for s in 0..s_total {
            for dd in 0..d {
                m = m.max((out[i] - expected(h, s, dd)).abs());
                i += 1;
            }
        }
    }
    m
}

#[allow(unsafe_code)]
#[allow(
    clippy::expect_used,
    reason = "test: array construction from a fixed in-bounds buffer cannot fail"
)]
fn f32_array(vals: &[f32], shape: &[i32]) -> Array {
    // SAFETY: f32 is 4-byte LE; from_bytes copies immediately.
    let bytes = unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("f32_array")
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: array/append construction from in-bounds fixed buffers cannot fail"
)]
fn cpu_two_append_multi_head_roundtrip() {
    let (kv_h, d) = (3, 32);
    let mut qk = new_turbo4(kv_h, d);
    let c0 = head_major_chunk(kv_h, 2, d, 0);
    let c1 = head_major_chunk(kv_h, 1, d, 2);
    let dummy = Array::from_bytes(&0.0_f32.to_le_bytes(), &[1], Dtype::F32).expect("dummy");
    qk.append(&c0, &[1, kv_h, 2, d], &dummy, Device::Cpu, 512)
        .expect("append0");
    qk.append(&c1, &[1, kv_h, 1, d], &dummy, Device::Cpu, 512)
        .expect("append1");
    let (out, arr) = qk
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");
    assert!(arr.is_none(), "CPU dequant returns a flat vec");
    let m = check(&out, kv_h, 3, d);
    assert!(
        m < 0.05,
        "turbo4 kv_h=3 two-append max abs error {m} — expected q4 noise, not head scramble"
    );
}

#[test]
#[ignore = "GPU Metal context — run explicitly: -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn gpu_two_append_multi_head_roundtrip() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d) = (2, 32);
    let mut qk = new_turbo4(kv_h, d);
    let c0 = head_major_chunk(kv_h, 2, d, 0);
    let c1 = head_major_chunk(kv_h, 1, d, 2);
    qk.append(
        &[],
        &[1, kv_h, 2, d],
        &f32_array(&c0, &[1, kv_h, 2, d]),
        Device::Gpu,
        512,
    )
    .expect("append0");
    qk.append(
        &[],
        &[1, kv_h, 1, d],
        &f32_array(&c1, &[1, kv_h, 1, d]),
        Device::Gpu,
        512,
    )
    .expect("append1");
    let (_, gpu) = qk
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("dequant");
    let gpu = gpu.expect("gpu array");
    gpu.eval().expect("eval");
    let bytes = gpu.to_bytes().expect("to_bytes");
    let out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk")))
        .collect();
    let m = check(&out, kv_h, 3, d);
    assert!(
        m < 0.05,
        "turbo4 GPU kv_h=2 two-append max abs error {m} — expected q4 noise, not head scramble"
    );
}
