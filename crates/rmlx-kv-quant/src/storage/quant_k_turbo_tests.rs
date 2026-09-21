//! Unit tests for [`QuantKTurbo`], at both code widths.
//!
//! The cases the two widths share are one generic body plus one `#[test]` per
//! width. The width-specific expectations a body cannot derive — the cosine
//! floor — arrive as parameters.
//!
//! # Test inventory
//!
//! | Body | What it holds |
//! |------|---------------|
//! | `new_shapes_correct` | shape / max_seq / bits wired by `new` |
//! | `roundtrip_cpu_single_step` | one append + dequant matches the reference codec |
//! | `append_cpu_path` | multi-step append accumulates seq |
//! | `reset_clears_seq` | reset zeroes shape + blocks |
//! | `cosine_empirical_floor_head_dim_128` | cosine ≥ the width's empirical floor |
//! | `from_cpu_blocks_max_seq_explicit` | `max_seq` is the explicit argument |
//! | `two_append_multi_head_roundtrip` | no head scramble across appends, CPU |
//! | `gpu_two_append_multi_head_roundtrip` | the same on the GPU append path |
//! | `cpu_msl_parity` | the scalar codec and the MSL kernel agree |
//! | `two_block_decode_matches_one_block_at_b_gt_1` | block boundary at `B > 1` |
//!
//! The three GPU bodies are `#[ignore]`-gated at both widths: they need a
//! Metal context and are excluded from a default `cargo test`.

use rmlx_mlx::{Array, Device, Dtype};

use crate::storage::quant_k_turbo::QuantKTurbo;
use crate::test_utils::{cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env, TEST_SEED};
use crate::turboquant::{turbo_dequantize, turbo_quantize_v, TurboBlocks};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a 1-element `Array` for use as the `k_arr` argument on the CPU path.
/// The CPU branch ignores `k_arr`; we pass a minimal array so the type is
/// satisfied without GPU allocation.
fn dummy_k_arr() -> Array {
    let bytes: [u8; 4] = 0.0_f32.to_le_bytes();
    Array::from_bytes(&bytes, &[1], Dtype::F32).expect("dummy_k_arr: from_bytes must succeed")
}

/// Append one decode step on the CPU path (no GPU context required).
fn cpu_append<const BITS: u8>(qk: &mut QuantKTurbo<BITS>, data: &[f32], new_shape: &[i32]) {
    let k_arr = dummy_k_arr();
    let n_seq = new_shape[2];
    qk.append(data, new_shape, &k_arr, Device::Cpu, n_seq)
        .expect("CPU append must succeed");
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

// ── Structure ────────────────────────────────────────────────────────────────

fn new_shapes_correct<const BITS: u8>() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantKTurbo::<BITS>::new(init_shape.clone(), max_seq);
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.max_seq, max_seq, "max_seq preserved");
    assert_eq!(q.bits, BITS, "bits must be the store's own width");
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert_eq!(q.byte_size(), 0, "byte_size 0 with no blocks");
    assert!(q.gpu_codes_buf.is_none(), "no GPU codes buf before append");
    assert!(
        q.gpu_scales_buf.is_none(),
        "no GPU scales buf before append"
    );
}

#[test]
fn quant_k_turbo3_new_shapes_correct() {
    new_shapes_correct::<3>();
}

#[test]
fn quant_k_turbo4_new_shapes_correct() {
    new_shapes_correct::<4>();
}

// ── CPU roundtrip ─────────────────────────────────────────────────────────────

/// Append one decode step (n_seq rows) and dequant on CPU; output must match
/// the reference `turbo_quantize_v` / `turbo_dequantize` directly.
fn roundtrip_cpu_single_step<const BITS: u8>() {
    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 32; // exactly one group per row
    let n_elems = b * kv_h * n_seq * head_dim;
    let data = lcg_data(n_elems, TEST_SEED);
    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantKTurbo::<BITS>::new(
        vec![b as i32, kv_h as i32, 0_i32, head_dim as i32],
        n_seq as i32,
    );
    cpu_append(&mut qk, &data, &new_shape);

    assert_eq!(qk.blocks.len(), 1, "one block after single append");
    assert_eq!(qk.shape[2], n_seq as i32, "seq dim updated");
    assert!(qk.byte_size() > 0, "byte_size non-zero after append");

    let decoded = qk.dequant().expect("dequant must succeed");

    // Reference: direct CPU encode/decode
    let ref_blocks =
        turbo_quantize_v(&data, BITS, &new_shape).expect("reference encode must succeed");
    let reference = turbo_dequantize(&ref_blocks).expect("reference decode must succeed");

    assert_eq!(decoded.len(), reference.len(), "output length mismatch");
    let max_abs_err = decoded
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err < 1e-6,
        "CPU roundtrip max_abs_err = {max_abs_err:.2e} (>= 1e-6)"
    );
}

#[test]
fn quant_k_turbo3_roundtrip_cpu_single_step() {
    roundtrip_cpu_single_step::<3>();
}

#[test]
fn quant_k_turbo4_roundtrip_cpu_single_step() {
    roundtrip_cpu_single_step::<4>();
}

// ── Multi-step append ─────────────────────────────────────────────────────────

/// Each call to `append` accumulates seq tokens; byte_size grows with steps.
fn append_cpu_path<const BITS: u8>() {
    let head_dim = 32; // one group per row
    let n_seq = 2;

    let data1 = lcg_data(n_seq * head_dim, TEST_SEED);
    let data2 = lcg_data(n_seq * head_dim, TEST_SEED.wrapping_add(1));
    let shape1 = [1_i32, 1, n_seq as i32, head_dim as i32];
    let shape2 = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantKTurbo::<BITS>::new(vec![1, 1, 0, head_dim as i32], 16);
    cpu_append(&mut qk, &data1, &shape1);
    assert_eq!(qk.shape[2], n_seq as i32);
    assert_eq!(qk.blocks.len(), 1);
    let size_after_first = qk.byte_size();

    cpu_append(&mut qk, &data2, &shape2);
    assert_eq!(qk.shape[2], 2 * n_seq as i32);
    assert_eq!(qk.blocks.len(), 2);
    assert!(
        qk.byte_size() > size_after_first,
        "byte_size must grow after second append"
    );
}

#[test]
fn quant_k_turbo3_append_cpu_path() {
    append_cpu_path::<3>();
}

#[test]
fn quant_k_turbo4_append_cpu_path() {
    append_cpu_path::<4>();
}

// ── Reset ─────────────────────────────────────────────────────────────────────

fn reset_clears_seq<const BITS: u8>() {
    let head_dim = 32;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantKTurbo::<BITS>::new(vec![1, 1, 0, head_dim as i32], 16);
    cpu_append(&mut qk, &data, &new_shape);
    assert_eq!(qk.shape[2], n_seq as i32);

    qk.reset();
    assert_eq!(qk.shape[2], 0, "seq dim must be 0 after reset");
    assert!(qk.blocks.is_empty(), "blocks must be empty after reset");
    assert_eq!(qk.byte_size(), 0);
}

#[test]
fn quant_k_turbo3_reset_clears_seq() {
    reset_clears_seq::<3>();
}

#[test]
fn quant_k_turbo4_reset_clears_seq() {
    reset_clears_seq::<4>();
}

// ── Cosine empirical floor ────────────────────────────────────────────────────

/// Cosine similarity gate for the turbo K codec at head_dim=128.
///
/// The codec is axis-agnostic — the K side uses the same Lloyd-Max codebook as
/// the V side at the same width. `floor` is a parameter because the two widths
/// do not share one: it is measured at this seed and shape, then gated at
/// measured − 0.001.
fn cosine_empirical_floor_head_dim_128<const BITS: u8>(floor: f32) {
    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantKTurbo::<BITS>::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    cpu_append(&mut qk, &data, &new_shape);
    let decoded = qk.dequant().expect("dequant must succeed");

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    assert!(
        stats.min >= floor,
        "turbo{BITS}_k cosine min={:.6} below empirical floor {floor} (mean={:.6}, n={})",
        stats.min,
        stats.mean,
        stats.n_rows,
    );
}

#[test]
fn quant_k_turbo3_cosine_empirical_floor_head_dim_128() {
    // V-side turbo3 anchor (quant_v_tests, turboquant row) is 0.9817; the K
    // side runs the same codec, so the gate is that anchor minus 0.001.
    cosine_empirical_floor_head_dim_128::<3>(0.9807);
}

#[test]
fn quant_k_turbo4_cosine_empirical_floor_head_dim_128() {
    // 4-bit is higher fidelity than 3-bit; measured min at this seed and shape
    // is 0.996401, gated at measured − 0.001.
    cosine_empirical_floor_head_dim_128::<4>(0.9954);
}

// ── from_cpu_blocks takes explicit max_seq ────────────────────────────────────

/// `from_cpu_blocks` must accept an explicit `max_seq` argument: the
/// constructor param is mandatory, not inferred from block count, to keep
/// capacity management correct after SSD hydrate.
fn from_cpu_blocks_max_seq_explicit<const BITS: u8>() {
    let head_dim = 32;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let block: TurboBlocks = turbo_quantize_v(&data, BITS, &shape).expect("encode must succeed");
    let blocks: Vec<TurboBlocks> = vec![block];
    let explicit_max_seq = 128_i32;

    let qk = QuantKTurbo::<BITS>::from_cpu_blocks(blocks, shape.to_vec(), explicit_max_seq);

    // max_seq must equal the explicit argument, NOT n_seq or inferred from block count
    assert_eq!(
        qk.max_seq, explicit_max_seq,
        "max_seq must be the explicit argument ({explicit_max_seq}), not {n_seq}"
    );
    assert_eq!(qk.bits, BITS);
    assert_eq!(qk.blocks.len(), 1);

    // Dequant must still work after from_cpu_blocks
    let decoded = qk.dequant().expect("dequant must succeed");
    assert_eq!(decoded.len(), data.len(), "decoded length must match input");
}

#[test]
fn quant_k_turbo3_from_cpu_blocks_max_seq_explicit() {
    from_cpu_blocks_max_seq_explicit::<3>();
}

#[test]
fn quant_k_turbo4_from_cpu_blocks_max_seq_explicit() {
    from_cpu_blocks_max_seq_explicit::<4>();
}

// ── Multi-append GQA layout round-trip ───────────────────────────────────────

/// Distinct, small per-(head,token,dim) value so a head transposition (which
/// swaps in a value differing by ≥ ~0.1) is obvious against quantizer noise.
fn rt_expected(h: i32, s: i32, d: i32) -> f32 {
    (h * 100 + s * 5 + d % 7) as f32 * 0.001
}

/// Head-major flat `[1, kv_h, seq, d]` chunk — the layout `append` receives.
fn rt_head_major_chunk(kv_h: i32, seq: i32, d: i32, base_s: i32) -> Vec<f32> {
    let mut v = Vec::with_capacity((kv_h * seq * d) as usize);
    for h in 0..kv_h {
        for s in 0..seq {
            for dd in 0..d {
                v.push(rt_expected(h, base_s + s, dd));
            }
        }
    }
    v
}

fn rt_check(out: &[f32], kv_h: i32, s_total: i32, d: i32) -> f32 {
    let mut m = 0.0_f32;
    let mut i = 0usize;
    for h in 0..kv_h {
        for s in 0..s_total {
            for dd in 0..d {
                m = m.max((out[i] - rt_expected(h, s, dd)).abs());
                i += 1;
            }
        }
    }
    m
}

/// Two head-major appends, kv_h=3: the pre-fix head-major store + head-major
/// reshape scrambled heads across the two blocks. The seq-major reorder fixes
/// it; max-err must be quantizer noise, not a head swap.
fn two_append_multi_head_roundtrip<const BITS: u8>() {
    let (kv_h, d) = (3, 32);
    let mut qk = QuantKTurbo::<BITS>::new(vec![1, kv_h, 0, d], 512);
    let c0 = rt_head_major_chunk(kv_h, 2, d, 0);
    let c1 = rt_head_major_chunk(kv_h, 1, d, 2);
    cpu_append(&mut qk, &c0, &[1, kv_h, 2, d]);
    cpu_append(&mut qk, &c1, &[1, kv_h, 1, d]);
    let out = qk.dequant().expect("dequant");
    let m = rt_check(&out, kv_h, 3, d);
    assert!(
        m < 0.05,
        "turbo{BITS} kv_h=3 two-append max abs error {m} — expected quantizer noise, not head scramble"
    );
}

#[test]
fn quant_k_turbo3_two_append_multi_head_roundtrip() {
    two_append_multi_head_roundtrip::<3>();
}

#[test]
fn quant_k_turbo4_two_append_multi_head_roundtrip() {
    two_append_multi_head_roundtrip::<4>();
}

/// GPU two-append multi-head round-trip — the path the layout bug lived on.
#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn gpu_two_append_multi_head_roundtrip<const BITS: u8>() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d) = (2, 32);
    let mut qk = QuantKTurbo::<BITS>::new(vec![1, kv_h, 0, d], 512);
    let c0 = rt_head_major_chunk(kv_h, 2, d, 0);
    let c1 = rt_head_major_chunk(kv_h, 1, d, 2);
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
    let m = rt_check(&out, kv_h, 3, d);
    assert!(
        m < 0.05,
        "turbo{BITS} GPU kv_h=2 two-append max abs error {m} — expected quantizer noise, not head scramble"
    );
}

#[test]
#[ignore = "GPU Metal context — run explicitly: -- --ignored --test-threads=1"]
fn quant_k_turbo3_gpu_two_append_multi_head_roundtrip() {
    gpu_two_append_multi_head_roundtrip::<3>();
}

#[test]
#[ignore = "GPU Metal context — run explicitly: -- --ignored --test-threads=1"]
fn quant_k_turbo4_gpu_two_append_multi_head_roundtrip() {
    gpu_two_append_multi_head_roundtrip::<4>();
}

// ── Parity: CPU == MSL (ignored, requires Metal) ─────────────────────────────

/// Parity body: CPU path vs MSL kernel — both must decode to within 1e-5.
///
/// The GPU path quantizes via this width's `turbo_quantize_v*_gpu` MSL kernel;
/// the CPU path uses the Rust scalar `turbo_quantize_v`. Both pack to the same
/// Lloyd-Max codebook, so decoded output must match within float rounding.
///
/// Skips silently when `RMLX_SKIP_GPU=1` (CI without Metal).
fn cpu_msl_parity<const BITS: u8>() {
    if skip_if_no_gpu_env() {
        return;
    }

    let head_dim = 128;
    let n_rows = 8;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    // CPU path
    let mut qk_cpu = QuantKTurbo::<BITS>::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    cpu_append(&mut qk_cpu, &data, &new_shape);
    let cpu_out = qk_cpu.dequant().expect("cpu dequant must succeed");

    // GPU path: pass the real Array so the GPU kernel has data
    let k_arr = f32_array(&data, &new_shape);
    let mut qk_gpu = QuantKTurbo::<BITS>::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    qk_gpu
        .append(&data, &new_shape, &k_arr, Device::Gpu, n_rows as i32)
        .expect("GPU append must succeed");
    let (gpu_out, gpu_arr_opt) = qk_gpu
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("GPU dequant must succeed");

    // The GPU path returns an Array; the dequant kernel's output is already
    // materialized, so `to_bytes()` copies the raw f32 bytes out.
    let gpu_vec: Vec<f32> = match gpu_arr_opt {
        Some(gpu_arr) => gpu_arr
            .to_bytes()
            .expect("GPU array to_bytes must succeed")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("chunk")))
            .collect(),
        None => gpu_out,
    };

    assert_eq!(
        cpu_out.len(),
        gpu_vec.len(),
        "CPU/GPU output length mismatch"
    );

    let max_err = cpu_out
        .iter()
        .zip(gpu_vec.iter())
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    // The codebook is hardwired identically in Rust and MSL; any non-trivial
    // deviation is a codec discrepancy.
    assert!(
        max_err < 1e-5,
        "turbo{BITS} CPU vs MSL max-abs-error = {max_err:.2e} exceeds 1e-5 (codec discrepancy)"
    );
}

#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn quant_k_turbo3_cpu_msl_parity() {
    cpu_msl_parity::<3>();
}

#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn quant_k_turbo4_cpu_msl_parity() {
    cpu_msl_parity::<4>();
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
/// concatenation back in `QuantKTurbo::dequantize_choice` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
fn two_block_decode_matches_one_block_at_b_gt_1<const BITS: u8>() {
    for (b, kv_h) in [(1_usize, 1_usize), (1, 2), (2, 1), (2, 2)] {
        let head_dim = 32_usize;
        let (n0, n1) = (2_usize, 3_usize);
        let max_seq = 512_i32;
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];
        let dummy =
            |n: usize| rmlx_mlx::zeros(&shape(n), Dtype::F32, Device::Cpu).expect("dummy array");
        let cpu_dequant = |st: &QuantKTurbo<BITS>| {
            st.dequantize_choice(Device::Cpu, Dtype::F32)
                .expect("cpu dequant")
                .0
        };

        let mut one =
            QuantKTurbo::<BITS>::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], max_seq);
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
            QuantKTurbo::<BITS>::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], max_seq);
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
            "two-block decode must equal the one-block oracle at b={b} kv_h={kv_h}"
        );
    }
}

#[test]
fn quant_k_turbo3_two_block_decode_matches_one_block_at_b_gt_1() {
    two_block_decode_matches_one_block_at_b_gt_1::<3>();
}

#[test]
fn quant_k_turbo4_two_block_decode_matches_one_block_at_b_gt_1() {
    two_block_decode_matches_one_block_at_b_gt_1::<4>();
}
