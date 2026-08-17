//! Tests for `QuantV` — codebook override surface + sequence-major layout
//! round-trips.
//!
//! Like `QuantK`, the `QuantV` flat GPU buffer accumulates one chunk per
//! `append` at a sequence offset (`prev_seq * words_per_seq`) while the dequant
//! reshapes the flat prefix as the logical `[B, kv_h, S, D]`. The fix stores
//! every chunk sequence-major (reorder heads↔seq before quantizing, reorder
//! back on dequant). The round-trip tests below prove no head scramble on a
//! multi-append GQA cache; the GPU ones are `#[ignore]` because they touch
//! Metal — run with `--ignored --test-threads=1`.

use super::QuantV;
use crate::test_utils::skip_if_no_gpu_env;
use rmlx_mlx::{zeros, Array, Device, Dtype};

// ── value_codebook field ──────────────────────────────────────────────────────

/// `QuantV::from_cpu_blocks` must initialise `value_codebook` to `None`.
#[test]
fn from_cpu_blocks_value_codebook_is_none() {
    let qv = QuantV::from_cpu_blocks(Vec::new(), vec![1, 1, 0, 32], 4);
    assert!(
        qv.value_codebook.is_none(),
        "from_cpu_blocks must start with value_codebook = None"
    );
}

/// Constructing `QuantV` with `value_codebook = Some(...)` stores the
/// codebook on the struct — the field is accessible after construction.
#[test]
fn value_codebook_stored_after_construction() {
    let codebook = vec![
        -2.717_667_f32,
        -2.052_138,
        -1.600_802_4,
        -1.239_959,
        -0.928_244_7,
        -0.645_875_33,
        -0.381_178_23,
        -0.126_046_94,
        0.126_046_94,
        0.381_178_23,
        0.645_875_33,
        0.928_244_7,
        1.239_959,
        1.600_802_4,
        2.052_138,
        2.717_667,
    ];
    let qv = QuantV {
        blocks: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: vec![1, 1, 0, 32],
        bits: 4,
        max_seq: 0,
        high_precision_indices: None,
        value_codebook: Some(codebook.clone()),
        value_codebook_gpu: None,
        use_tcq: false,
    };
    assert_eq!(
        qv.value_codebook.as_deref(),
        Some(codebook.as_slice()),
        "value_codebook must be stored exactly as provided"
    );
}

/// `try_deep_clone` must propagate `value_codebook`.
#[test]
fn try_deep_clone_propagates_value_codebook() {
    let cb = vec![-1.5_f32, -0.5, 0.5, 1.5];
    let qv = QuantV {
        blocks: Vec::new(),
        gpu_codes_buf: None,
        gpu_scales_buf: None,
        gpu_words_per_step: 0,
        gpu_scales_per_step: 0,
        gpu_capacity: 0,
        shape: vec![1, 1, 0, 32],
        bits: 2,
        max_seq: 0,
        high_precision_indices: None,
        value_codebook: Some(cb.clone()),
        value_codebook_gpu: None,
        use_tcq: false,
    };
    let cloned = qv.try_deep_clone().expect("try_deep_clone must succeed");
    assert_eq!(
        cloned.value_codebook.as_deref(),
        Some(cb.as_slice()),
        "try_deep_clone must propagate value_codebook"
    );
}

// ── Sequence-major layout round-trip ─────────────────────────────────────────

/// Distinct, small per-(head,token,dim) value: q8/Lloyd-Max noise stays ≪ 0.01
/// while a head transposition (swaps in a value differing by ≥ ~0.1) is obvious.
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

fn new_quant_v(kv_h: i32, d: i32, bits: u8) -> QuantV {
    QuantV::new_affine_decode(vec![1, kv_h, 0, d], bits, 512)
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
#[allow(
    clippy::expect_used,
    reason = "test: array/append construction from in-bounds fixed buffers cannot fail"
)]
fn cpu_two_append_multi_head_roundtrip_is_exact() {
    // CPU path (bits=4): two head-major appends, kv_h=2. The pre-fix head-major
    // store + head-major reshape scrambled heads across the two blocks.
    let (kv_h, d) = (2, 64);
    let mut qv = new_quant_v(kv_h, d, 4);
    let c0 = head_major_chunk(kv_h, 2, d, 0);
    let c1 = head_major_chunk(kv_h, 1, d, 2);
    let dummy0 = zeros(&[1, kv_h, 2, d], Dtype::F32, Device::Cpu).expect("dummy0");
    let dummy1 = zeros(&[1, kv_h, 1, d], Dtype::F32, Device::Cpu).expect("dummy1");
    qv.append(&c0, &[1, kv_h, 2, d], &dummy0, Device::Cpu, 512)
        .expect("append0");
    qv.append(&c1, &[1, kv_h, 1, d], &dummy1, Device::Cpu, 512)
        .expect("append1");
    let (flat, arr) = qv
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");
    assert!(arr.is_none(), "CPU dequant returns a flat vec");
    let m = check_roundtrip(&flat, kv_h, 3, d);
    assert!(
        m < 0.05,
        "CPU kv_h=2 two-append max abs error {m} — expected quant noise, not head scramble"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: array/append construction from in-bounds fixed buffers cannot fail"
)]
fn cpu_cross_head_group_roundtrip() {
    // TurboQuant group size is 32; head_dim=32, kv_h=3, per-token decode steps.
    // A single decode chunk is kv_h*d = 96 elems = 3 groups, one per head. Two
    // appends land the second chunk at a sequence offset, exercising the
    // multi-append reorder.
    let (kv_h, d) = (3, 32);
    let mut qv = new_quant_v(kv_h, d, 4);
    let c0 = head_major_chunk(kv_h, 1, d, 0);
    let c1 = head_major_chunk(kv_h, 1, d, 1);
    let dummy = zeros(&[1, kv_h, 1, d], Dtype::F32, Device::Cpu).expect("dummy");
    qv.append(&c0, &[1, kv_h, 1, d], &dummy, Device::Cpu, 512)
        .expect("append0");
    qv.append(&c1, &[1, kv_h, 1, d], &dummy, Device::Cpu, 512)
        .expect("append1");
    let (flat, _) = qv
        .dequantize_choice(Device::Cpu, Dtype::F32)
        .expect("dequant");
    let m = check_roundtrip(&flat, kv_h, 2, d);
    assert!(
        m < 0.05,
        "CPU kv_h=3 cross-group max abs error {m} — expected quant noise, not head scramble"
    );
}

// ── GPU round-trip (real QuantV + Metal kernels) ─────────────────────────────

#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn make_v_array(kv_h: i32, seq: i32, d: i32, base_s: i32) -> Array {
    let data = head_major_chunk(kv_h, seq, d, base_s);
    // SAFETY: Apple-Silicon-only build; f32 is 4-byte LE. `data` is borrowed
    // read-only and copied into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, &[1, kv_h, seq, d], Dtype::F32).expect("make_v_array")
}

#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn append_gpu(qv: &mut QuantV, kv_h: i32, seq: i32, d: i32, base_s: i32) {
    let arr = make_v_array(kv_h, seq, d, base_s);
    arr.eval().expect("eval v");
    qv.append(&[], &[1, kv_h, seq, d], &arr, Device::Gpu, 512)
        .expect("append");
}

#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn dequant_to_vec(qv: &QuantV) -> Vec<f32> {
    let (_, out) = qv
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

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_v -- --ignored --test-threads=1"]
fn gpu_two_append_multi_head_roundtrip() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d) = (2, 64);
    let mut qv = new_quant_v(kv_h, d, 4);
    append_gpu(&mut qv, kv_h, 2, d, 0);
    append_gpu(&mut qv, kv_h, 1, d, 2);
    let out = dequant_to_vec(&qv);
    let m = check_roundtrip(&out, kv_h, 3, d);
    assert!(
        m < 0.05,
        "kv_h=2 two-append max abs error {m} — expected quant noise, not head scramble"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_v -- --ignored --test-threads=1"]
fn gpu_two_append_single_head_control() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d) = (1, 64);
    let mut qv = new_quant_v(kv_h, d, 4);
    append_gpu(&mut qv, kv_h, 2, d, 0);
    append_gpu(&mut qv, kv_h, 1, d, 2);
    let out = dequant_to_vec(&qv);
    let m = check_roundtrip(&out, kv_h, 3, d);
    assert!(m < 0.05, "kv_h=1 control max abs error {m}");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test quant_v -- --ignored --test-threads=1"]
fn gpu_single_shot_cold_prefill() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d) = (4, 64);
    let mut qv = new_quant_v(kv_h, d, 4);
    append_gpu(&mut qv, kv_h, 8, d, 0);
    let out = dequant_to_vec(&qv);
    let m = check_roundtrip(&out, kv_h, 8, d);
    assert!(m < 0.05, "single-shot cold prefill max abs error {m}");
}

// ── Batch-axis block-boundary parity ──────────────────────────────────

/// Two appends must decode exactly like one append of the same tokens, at
/// `B > 1` as well as `B == 1`.
///
/// Each block covers `[B, S_block, kv_h, D]`, so the concatenation of two
/// blocks is not one `[B, S_total, kv_h, D]` run — reading it as one maps the
/// second block's batch-0 rows onto batch-1 sequence slots. The single-append
/// store holds exactly one block and therefore concatenates nothing, which is
/// what makes it the oracle here.
///
/// Mutation check: put `seq_layout::transpose_seq_heads` over the whole
/// concatenation back in `QuantV::dequantize_choice` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_v_two_block_decode_matches_one_block_at_b_gt_1() {
    for b in [1_usize, 2] {
        let (kv_h, head_dim) = (2_usize, 32_usize);
        let (n0, n1) = (2_usize, 3_usize);
        let max_seq = 512_i32;
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];
        let dummy = |n: usize| zeros(&shape(n), Dtype::F32, Device::Cpu).expect("dummy array");
        let cpu_dequant = |st: &QuantV| {
            st.dequantize_choice(Device::Cpu, Dtype::F32)
                .expect("cpu dequant")
                .0
        };

        let mut one =
            QuantV::new_affine_decode(vec![b as i32, kv_h as i32, 0, head_dim as i32], 4, max_seq);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
            &dummy(n0 + n1),
            Device::Cpu,
            max_seq,
        )
        .expect("single append");
        let oracle = cpu_dequant(&one);

        let mut two =
            QuantV::new_affine_decode(vec![b as i32, kv_h as i32, 0, head_dim as i32], 4, max_seq);
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0, head_dim),
            &shape(n0),
            &dummy(n0),
            Device::Cpu,
            max_seq,
        )
        .expect("append chunk 0");
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, n0, n1, head_dim),
            &shape(n1),
            &dummy(n1),
            Device::Cpu,
            max_seq,
        )
        .expect("append chunk 1");
        let got = cpu_dequant(&two);

        assert_eq!(
            got, oracle,
            "two-block decode must equal the one-block oracle at b={b}"
        );
    }
}
