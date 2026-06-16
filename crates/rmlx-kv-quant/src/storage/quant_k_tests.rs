//! Layout round-trip tests for `QuantK` (q8_0 K storage).
//!
//! The GPU K buffer accumulates one chunk per `append` at a sequence offset
//! (`prev_seq * words_per_seq`). The dequant view must read that flat buffer
//! back as the logical `[B, kv_h, S, D]` shape. The two only agree when the
//! storage and the read use the **same** ordering of the (head, token) axes.
//!
//! Historically the chunk was stored head-major (`[B, kv_h, new_seq, D]`) but
//! read with a `[B, kv_h, S, D]` reshape of the flat buffer. With a single
//! chunk those coincide, so cold-prefill was correct; with two or more chunks
//! and `kv_h > 1` the second chunk's tokens landed after *all* heads' prefixes
//! while the reshape mapped one head's new-token slot onto another head's
//! prefix — a head transposition that silently corrupted K.
//!
//! The fix makes the buffer uniformly sequence-major: `append` transposes the
//! chunk heads↔seq before quantizing, and `dequantize_choice` reshapes the flat
//! prefix as `[B, S, kv_h, D]` and transposes back to `[B, kv_h, S, D]`.
//!
//! The deterministic index-math tests below model the GPU offset arithmetic on
//! the CPU (no Metal context) and prove the fail-before / pass-after behaviour.
//! The GPU round-trip tests drive the real `QuantK` and are `#[ignore]` because
//! they touch Metal — run with `--ignored --test-threads=1`.

use super::QuantK;
use crate::test_utils::skip_if_no_gpu_env;
use rmlx_mlx::{zeros, Array, Device, Dtype};

// ── Deterministic index-math model (no GPU) ──────────────────────────────────
//
// We model the flat code buffer as one logical element per (b, h, s, d). The
// q8 packing (4 codes per u32) and grouping (128 elems per scale) are a fixed
// bijection *within* a contiguous run, so they do not change which logical
// element lands at which buffer slot — only the per-chunk ordering and the
// per-append offset do. Those are exactly what this model captures.

const B: usize = 1;

/// Distinct value per (head, token, dim) so any transposition is detectable.
fn val(h: usize, s: usize, d: usize) -> f32 {
    (h * 100_000 + s * 100 + d) as f32
}

/// Build the flat buffer the way `QuantK::append` does, for a list of chunk
/// lengths. `head_major_chunk = true` is the pre-fix ordering (chunk stored
/// `[b][h][s][d]`); `false` is the fixed ordering (chunk stored `[b][s][h][d]`).
fn build_buffer(appends: &[usize], kv_h: usize, d: usize, head_major_chunk: bool) -> Vec<f32> {
    let elems_per_seq = B * kv_h * d;
    let total: usize = appends.iter().sum::<usize>() * elems_per_seq;
    let mut buf = vec![f32::NAN; total];
    let mut prev = 0usize;
    for &new_seq in appends {
        let start = prev * elems_per_seq;
        let mut i = start;
        if head_major_chunk {
            for _b in 0..B {
                for h in 0..kv_h {
                    for sl in 0..new_seq {
                        for dd in 0..d {
                            buf[i] = val(h, prev + sl, dd);
                            i += 1;
                        }
                    }
                }
            }
        } else {
            for _b in 0..B {
                for sl in 0..new_seq {
                    for h in 0..kv_h {
                        for dd in 0..d {
                            buf[i] = val(h, prev + sl, dd);
                            i += 1;
                        }
                    }
                }
            }
        }
        prev += new_seq;
    }
    buf
}

/// Old (buggy) read: reshape the flat prefix directly to `[B, kv_h, S, D]`.
fn read_head_major(
    buf: &[f32],
    s_total: usize,
    _kv_h: usize,
    d: usize,
    h: usize,
    s: usize,
    dd: usize,
) -> f32 {
    let idx = ((h * s_total) + s) * d + dd;
    buf[idx]
}

/// New (fixed) read: reshape the flat prefix to `[B, S, kv_h, D]`, then
/// transpose heads↔seq to `[B, kv_h, S, D]`.
fn read_seq_major(
    buf: &[f32],
    _s_total: usize,
    kv_h: usize,
    d: usize,
    h: usize,
    s: usize,
    dd: usize,
) -> f32 {
    let idx = ((s * kv_h) + h) * d + dd;
    buf[idx]
}

fn max_err<R>(appends: &[usize], kv_h: usize, d: usize, head_major_chunk: bool, read: R) -> f32
where
    R: Fn(&[f32], usize, usize, usize, usize, usize, usize) -> f32,
{
    let buf = build_buffer(appends, kv_h, d, head_major_chunk);
    let s_total: usize = appends.iter().sum();
    let mut m = 0.0_f32;
    for h in 0..kv_h {
        for s in 0..s_total {
            for dd in 0..d {
                let got = read(&buf, s_total, kv_h, d, h, s, dd);
                m = m.max((got - val(h, s, dd)).abs());
            }
        }
    }
    m
}

#[test]
fn layout_bug_reproduces_for_multi_append_multi_head() {
    // Pre-fix path: head-major chunk store + head-major reshape read.
    // Single chunk: agrees (cold-prefill is correct).
    assert_eq!(
        max_err(&[3], 2, 4, true, read_head_major),
        0.0,
        "single-shot cold-prefill must be exact even on the buggy path"
    );
    // kv_h == 1 control: two appends still agree (no head axis to transpose).
    assert_eq!(
        max_err(&[2, 1], 1, 4, true, read_head_major),
        0.0,
        "kv_h=1 control must be exact"
    );
    // kv_h > 1 + two appends: head transposition corrupts the read.
    let bug = max_err(&[2, 1], 2, 4, true, read_head_major);
    assert!(
        bug > 0.0,
        "expected multi-head multi-append corruption on the pre-fix path, got {bug}"
    );
}

#[test]
fn layout_fix_is_exact_for_all_append_patterns() {
    // Fixed path: sequence-major chunk store + sequence-major reshape read.
    for &(appends, kv_h, d) in &[
        (&[3usize][..], 2usize, 4usize), // single-shot cold prefill
        (&[2, 1][..], 1, 4),             // kv_h = 1 control
        (&[2, 1][..], 2, 4),             // the bug case
        (&[2, 2][..], 3, 4),             // multi-head, even split
        (&[5, 3, 1][..], 4, 8),          // three appends, 4 heads
        (&[1, 1, 1, 1][..], 8, 4),       // per-token decode, 8 heads
    ] {
        let m = max_err(appends, kv_h, d, false, read_seq_major);
        assert_eq!(
            m, 0.0,
            "fixed path must be exact for appends={appends:?} kv_h={kv_h} d={d}, got {m}"
        );
    }
}

#[test]
fn cold_prefill_layout_is_byte_identical_after_fix() {
    // The single-chunk cold-prefill case: the fixed store+read must produce the
    // exact same per-(head,token) mapping as the pre-fix store+read did. (Both
    // are correct; this pins that the common path's logical contents are
    // unchanged — only the multi-append path's ordering is repaired.)
    for &(kv_h, d) in &[(1usize, 4usize), (2, 4), (4, 8), (8, 4)] {
        let pre = max_err(&[5], kv_h, d, true, read_head_major);
        let post = max_err(&[5], kv_h, d, false, read_seq_major);
        assert_eq!(pre, 0.0, "pre-fix cold prefill exact");
        assert_eq!(post, 0.0, "post-fix cold prefill exact");
    }
}

// ── Real CPU QuantK round-trip (no GPU, fully deterministic) ─────────────────

/// Head-major flat `[1, kv_h, seq, d]` chunk with distinct per-(head,token,dim)
/// values, the layout `QuantK::append` receives as `f32_data`.
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

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: array/append construction from in-bounds fixed buffers cannot fail"
)]
fn cpu_two_append_multi_head_roundtrip_is_exact() {
    // Drive the real `QuantK::append` + `dequantize_choice` on the CPU device.
    // The reorder is flat-position-symmetric on store and read, so the
    // round-trip is correct regardless of whether a q8 group crosses a
    // (head, token) boundary (see `cpu_gemma4_shape_cross_head_group_roundtrip`
    // for the d=64 boundary-crossing case). `d=128` here only because
    // `q8_quantize` asserts the chunk length is a multiple of the group size
    // (128) — with `d=128` and any `kv_h`, every chunk length `kv_h*d` divides
    // evenly.
    let (kv_h, d, max_seq) = (3, 128, 512);
    let mut qk = new_quant_k(kv_h, d);

    // Two appends: a 2-token prefix then a 1-token decode step.
    let c0 = head_major_chunk(kv_h, 2, d, 0);
    let c1 = head_major_chunk(kv_h, 1, d, 2);
    // CPU path ignores `k_arr`; pass a shape-correct dummy.
    let dummy0 = zeros(&[1, kv_h, 2, d], Dtype::F32, Device::Cpu).expect("dummy0");
    let dummy1 = zeros(&[1, kv_h, 1, d], Dtype::F32, Device::Cpu).expect("dummy1");
    qk.append(&c0, &[1, kv_h, 2, d], &dummy0, Device::Cpu, max_seq)
        .expect("append0");
    qk.append(&c1, &[1, kv_h, 1, d], &dummy1, Device::Cpu, max_seq)
        .expect("append1");

    let (flat, arr) = qk
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");
    assert!(arr.is_none(), "CPU dequant returns a flat vec");
    // `flat` is logical head-major [1, kv_h, S, D]; compare per (head, token).
    let s_total = 3;
    let m = check_roundtrip(&flat, kv_h, s_total, d);
    assert!(
        m < 0.02,
        "CPU kv_h=3 two-append max abs error {m} — expected quant noise, not head scramble"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: array/append construction from in-bounds fixed buffers cannot fail"
)]
fn cpu_gemma4_shape_cross_head_group_roundtrip() {
    // The production Gemma4 KV shape: head_dim=64, kv_h=2. A single-token decode
    // step produces a chunk of length kv_h*d = 2*64 = 128 — exactly one q8 group,
    // which therefore spans TWO heads. This is the configuration the sequence-
    // major reorder most needs to prove: if the per-(head,token) reorder were
    // wrong when a group crosses a head boundary, this test fails with an error
    // far above q8 noise (a whole head's values swapped in, ≥ ~0.1).
    let (kv_h, d, max_seq) = (2, 64, 512);
    let mut qk = new_quant_k(kv_h, d);

    // Two appends so the second chunk lands at a sequence offset (prev_seq=1),
    // exercising the multi-append reorder that the head-major store corrupted.
    let c0 = head_major_chunk(kv_h, 1, d, 0);
    let c1 = head_major_chunk(kv_h, 1, d, 1);
    let dummy0 = zeros(&[1, kv_h, 1, d], Dtype::F32, Device::Cpu).expect("dummy0");
    let dummy1 = zeros(&[1, kv_h, 1, d], Dtype::F32, Device::Cpu).expect("dummy1");
    qk.append(&c0, &[1, kv_h, 1, d], &dummy0, Device::Cpu, max_seq)
        .expect("append0");
    qk.append(&c1, &[1, kv_h, 1, d], &dummy1, Device::Cpu, max_seq)
        .expect("append1");

    let (flat, arr) = qk
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");
    assert!(arr.is_none(), "CPU dequant returns a flat vec");
    let s_total = 2;
    let m = check_roundtrip(&flat, kv_h, s_total, d);
    assert!(
        m < 0.02,
        "Gemma4 d=64/kv_h=2 cross-head-group max abs error {m} — expected q8 noise, not head scramble"
    );
}

// ── GPU round-trip tests (real QuantK + Metal kernels) ───────────────────────

fn make_k_array(kv_h: i32, seq: i32, d: i32, base_s: i32) -> Array {
    // Distinct value per (head, token, dim), shape [1, kv_h, seq, d].
    let n = (kv_h * seq * d) as usize;
    let mut data = vec![0.0_f32; n];
    let mut i = 0usize;
    for h in 0..kv_h {
        for s in 0..seq {
            for dd in 0..d {
                data[i] = expected(h, base_s + s, dd);
                i += 1;
            }
        }
    }
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and copied into MLX
    // before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    #[allow(
        clippy::expect_used,
        reason = "test helper: array construction from a fixed in-bounds buffer cannot fail"
    )]
    Array::from_bytes(bytes, &[1, kv_h, seq, d], Dtype::F32).expect("make_k_array")
}

#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn dequant_to_vec(qk: &QuantK) -> Vec<f32> {
    let (_, out) = qk
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("dequantize_choice");
    let out = out.expect("GPU dequant array");
    out.eval().expect("eval");
    let bytes = out.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk")))
        .collect()
}

#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn append_gpu(qk: &mut QuantK, kv_h: i32, seq: i32, d: i32, base_s: i32, max_seq: i32) {
    let arr = make_k_array(kv_h, seq, d, base_s);
    arr.eval().expect("eval k");
    let shape = [1, kv_h, seq, d];
    // CPU f32 data is unused on the GPU path; pass an empty slice.
    qk.append(&[], &shape, &arr, Device::Gpu, max_seq)
        .expect("append");
}

/// Expected per-(head,token,dim) value matching `make_k_array`.
///
/// Values are kept small (≲ 0.35) and distinct per (head, token) so that q8
/// quantization noise stays ≪ 0.01, while a head/token transposition (which
/// would swap in a value differing by ≥ ~0.1) is unmistakable.
fn expected(h: i32, s: i32, d: i32) -> f32 {
    (h * 100 + s * 5 + d % 7) as f32 * 0.001
}

fn new_quant_k(kv_h: i32, d: i32) -> QuantK {
    QuantK {
        codes: Vec::new(),
        scales: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: vec![1, kv_h, 0, d],
        max_seq: 0,
    }
}

fn check_roundtrip(out: &[f32], kv_h: i32, s_total: i32, d: i32) -> f32 {
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

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k -- --ignored --test-threads=1"]
fn gpu_two_append_multi_head_roundtrip() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d, max_seq) = (2, 128, 512);
    let mut qk = new_quant_k(kv_h, d);
    append_gpu(&mut qk, kv_h, 2, d, 0, max_seq);
    append_gpu(&mut qk, kv_h, 1, d, 2, max_seq);
    let out = dequant_to_vec(&qk);
    let m = check_roundtrip(&out, kv_h, 3, d);
    assert!(
        m < 0.02,
        "kv_h=2 two-append max abs error {m} — expected quant noise, not head scramble"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k -- --ignored --test-threads=1"]
fn gpu_two_append_single_head_control() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d, max_seq) = (1, 128, 512);
    let mut qk = new_quant_k(kv_h, d);
    append_gpu(&mut qk, kv_h, 2, d, 0, max_seq);
    append_gpu(&mut qk, kv_h, 1, d, 2, max_seq);
    let out = dequant_to_vec(&qk);
    let m = check_roundtrip(&out, kv_h, 3, d);
    assert!(m < 0.02, "kv_h=1 control max abs error {m}");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k -- --ignored --test-threads=1"]
fn gpu_single_shot_cold_prefill() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d, max_seq) = (4, 128, 512);
    let mut qk = new_quant_k(kv_h, d);
    append_gpu(&mut qk, kv_h, 8, d, 0, max_seq);
    let out = dequant_to_vec(&qk);
    let m = check_roundtrip(&out, kv_h, 8, d);
    assert!(m < 0.02, "single-shot cold prefill max abs error {m}");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_k -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn gpu_hydrate_then_decode_append_roundtrip() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Simulate the SSD hydrate path: build CPU codes/scales for a 2-token,
    // multi-head prefix using the CPU q8 encoder (sequence-major: the same
    // layout the spill path serializes), reconstruct a CPU-form QuantK via
    // `from_cpu_parts`, then drive a GPU decode append and dequant.
    let (kv_h, d, max_seq) = (2, 128, 512);
    let prefix_seq = 2;

    // CPU codes are produced from a sequence-major flat prefix [B, S, kv_h, D].
    let mut flat = Vec::with_capacity((prefix_seq * kv_h * d) as usize);
    for s in 0..prefix_seq {
        for h in 0..kv_h {
            for dd in 0..d {
                flat.push(expected(h, s, dd));
            }
        }
    }
    let (codes, scales) = crate::q8::q8_quantize(&flat);
    let mut qk = QuantK::from_cpu_parts(codes, scales, vec![1, kv_h, prefix_seq, d]);

    // First GPU append after hydrate triggers the buffer-init upload + a
    // sequence offset write — the exact path the bug lived on.
    append_gpu(&mut qk, kv_h, 1, d, prefix_seq, max_seq);

    let out = dequant_to_vec(&qk);
    let m = check_roundtrip(&out, kv_h, prefix_seq + 1, d);
    assert!(
        m < 0.02,
        "hydrate + decode-append max abs error {m} — expected quant noise, not scramble"
    );
}
