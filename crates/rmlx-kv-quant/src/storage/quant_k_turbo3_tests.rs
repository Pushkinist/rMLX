//! Unit tests for [`QuantKTurbo3`].
//!
//! # Test inventory
//!
//! | Test | DoD item |
//! |------|---------|
//! | `quant_k_turbo3_new_shapes_correct` | structure: shape/max_seq/bits wired |
//! | `quant_k_turbo3_roundtrip_cpu_single_step` | single append + dequant, CPU path |
//! | `quant_k_turbo3_append_cpu_path` | multi-step append accumulates seq |
//! | `quant_k_turbo3_reset_clears_seq` | reset zeroes shape + blocks |
//! | `quant_k_turbo3_cosine_empirical_floor_head_dim_128` | cosine ≥ empirical floor |
//! | `quant_k_turbo3_from_cpu_blocks_max_seq_explicit` | max_seq is explicit (not inferred) |
//!
//! The GPU parity test (`#[ignore]`-gated) lives in a separate `#[ignore]` test
//! below; it requires a Metal GPU context and is excluded from default `cargo test`.

use rmlx_mlx::{Array, Device, Dtype};

use crate::storage::quant_k_turbo3::{QuantKTurbo3, TURBO3_K_BITS};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env, TEST_SEED};
use crate::turboquant::{turbo_dequantize, turbo_quantize_v, TurboBlocks};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build a 1-element `Array` for use as the `k_arr` argument on the CPU path.
/// The CPU branch ignores `k_arr`; we pass a minimal array so the type is
/// satisfied without GPU allocation.
fn dummy_k_arr() -> Array {
    // SAFETY: `f32` is `Copy` with fixed 4-byte LE layout; `from_bytes` copies
    // immediately, so the temporary bytes array is not retained.
    let bytes: [u8; 4] = 0.0_f32.to_le_bytes();
    Array::from_bytes(&bytes, &[1], Dtype::F32).expect("dummy_k_arr: from_bytes must succeed")
}

/// Append one decode step on the CPU path (no GPU context required).
fn cpu_append(qk: &mut QuantKTurbo3, data: &[f32], new_shape: &[i32]) {
    let k_arr = dummy_k_arr();
    let n_seq = new_shape[2];
    qk.append(data, new_shape, &k_arr, Device::Cpu, n_seq)
        .expect("CPU append must succeed");
}

// ── Structure ────────────────────────────────────────────────────────────────

#[test]
fn quant_k_turbo3_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantKTurbo3::new(init_shape.clone(), max_seq);
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.max_seq, max_seq, "max_seq preserved");
    assert_eq!(q.bits, TURBO3_K_BITS, "bits must be TURBO3_K_BITS (3)");
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert_eq!(q.byte_size(), 0, "byte_size 0 with no blocks");
    assert!(q.gpu_codes_buf.is_none(), "no GPU codes buf before append");
    assert!(
        q.gpu_scales_buf.is_none(),
        "no GPU scales buf before append"
    );
}

// ── CPU roundtrip ─────────────────────────────────────────────────────────────

/// Append one decode step (n_seq rows) and dequant on CPU; output must match
/// the reference `turbo_quantize_v` / `turbo_dequantize` directly.
#[test]
fn quant_k_turbo3_roundtrip_cpu_single_step() {
    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 32; // exactly one group per row
    let n_elems = b * kv_h * n_seq * head_dim;
    let data = lcg_data(n_elems, TEST_SEED);
    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantKTurbo3::new(
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
        turbo_quantize_v(&data, TURBO3_K_BITS, &new_shape).expect("reference encode must succeed");
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

// ── Multi-step append ─────────────────────────────────────────────────────────

/// Each call to `append` accumulates seq tokens; byte_size grows with steps.
#[test]
fn quant_k_turbo3_append_cpu_path() {
    let head_dim = 32; // one group per row
    let n_seq = 2;

    let data1 = lcg_data(n_seq * head_dim, TEST_SEED);
    let data2 = lcg_data(n_seq * head_dim, TEST_SEED.wrapping_add(1));
    let shape1 = [1_i32, 1, n_seq as i32, head_dim as i32];
    let shape2 = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantKTurbo3::new(vec![1, 1, 0, head_dim as i32], 16);
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

// ── Reset ─────────────────────────────────────────────────────────────────────

#[test]
fn quant_k_turbo3_reset_clears_seq() {
    let head_dim = 32;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantKTurbo3::new(vec![1, 1, 0, head_dim as i32], 16);
    cpu_append(&mut qk, &data, &new_shape);
    assert_eq!(qk.shape[2], n_seq as i32);

    qk.reset();
    assert_eq!(qk.shape[2], 0, "seq dim must be 0 after reset");
    assert!(qk.blocks.is_empty(), "blocks must be empty after reset");
    assert_eq!(qk.byte_size(), 0);
}

// ── Cosine empirical floor ────────────────────────────────────────────────────

/// Cosine similarity gate for turbo3-K at head_dim=128.
///
/// The turbo3 codec is axis-agnostic — the K side uses the same Lloyd-Max
/// 3-bit codebook as the V side. The cosine floor is measured at the same
/// seed/shape and gated at measured − 0.001.
///
/// V-side empirical anchor (quant_v_tests, turboquant row): 0.9817.
/// K-side uses the same codec; floor set at 0.9807 (anchor − 0.001).
#[test]
fn quant_k_turbo3_cosine_empirical_floor_head_dim_128() {
    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantKTurbo3::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    cpu_append(&mut qk, &data, &new_shape);
    let decoded = qk.dequant().expect("dequant must succeed");

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    // Empirical floor: turbo3 Lloyd-Max 3-bit at D=128 on LCG-Gaussian data.
    // Gate at 0.9807 (V-side anchor 0.9817 minus 0.001).
    assert!(
        stats.min >= 0.9807,
        "turbo3_k cosine min={:.6} below empirical floor 0.9807 (mean={:.6}, n={})",
        stats.min,
        stats.mean,
        stats.n_rows,
    );
}

// ── from_cpu_blocks takes explicit max_seq ────────────────────────────────────

/// `from_cpu_blocks` must accept an explicit `max_seq` argument: the
/// constructor param is mandatory, not inferred from block count, to keep
/// capacity management correct after SSD hydrate.
#[test]
fn quant_k_turbo3_from_cpu_blocks_max_seq_explicit() {
    let head_dim = 32;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let block: TurboBlocks =
        turbo_quantize_v(&data, TURBO3_K_BITS, &shape).expect("encode must succeed");
    let blocks: Vec<TurboBlocks> = vec![block];
    let explicit_max_seq = 128_i32;

    let qk = QuantKTurbo3::from_cpu_blocks(blocks, shape.to_vec(), TURBO3_K_BITS, explicit_max_seq);

    // max_seq must equal the explicit argument, NOT n_seq or inferred from block count
    assert_eq!(
        qk.max_seq, explicit_max_seq,
        "max_seq must be the explicit argument ({explicit_max_seq}), not {n_seq}"
    );
    assert_eq!(qk.bits, TURBO3_K_BITS);
    assert_eq!(qk.blocks.len(), 1);

    // Dequant must still work after from_cpu_blocks
    let decoded = qk.dequant().expect("dequant must succeed");
    assert_eq!(decoded.len(), data.len(), "decoded length must match input");
}

// ── Multi-append GQA layout round-trip ───────────────────────────────────────

/// Distinct, small per-(head,token,dim) value so a head transposition (which
/// swaps in a value differing by ≥ ~0.1) is obvious against q3 noise.
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

#[allow(unsafe_code)]
#[allow(
    clippy::expect_used,
    reason = "test: array construction from a fixed in-bounds buffer cannot fail"
)]
fn rt_f32_array(vals: &[f32], shape: &[i32]) -> Array {
    // SAFETY: f32 is 4-byte LE; from_bytes copies immediately.
    let bytes = unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("rt_f32_array")
}

/// Two head-major appends, kv_h=3: the pre-fix head-major store + head-major
/// reshape scrambled heads across the two blocks. The seq-major reorder fixes
/// it; max-err must be q3 noise, not a head swap.
#[test]
fn quant_k_turbo3_two_append_multi_head_roundtrip() {
    let (kv_h, d) = (3, 32);
    let mut qk = QuantKTurbo3::new(vec![1, kv_h, 0, d], 512);
    let c0 = rt_head_major_chunk(kv_h, 2, d, 0);
    let c1 = rt_head_major_chunk(kv_h, 1, d, 2);
    cpu_append(&mut qk, &c0, &[1, kv_h, 2, d]);
    cpu_append(&mut qk, &c1, &[1, kv_h, 1, d]);
    let out = qk.dequant().expect("dequant");
    let m = rt_check(&out, kv_h, 3, d);
    assert!(
        m < 0.05,
        "turbo3 kv_h=3 two-append max abs error {m} — expected q3 noise, not head scramble"
    );
}

/// GPU two-append multi-head round-trip — the path the layout bug lived on.
#[test]
#[ignore = "GPU Metal context — run explicitly: -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test: structural invariant established by construction; .expect() documents it"
)]
fn quant_k_turbo3_gpu_two_append_multi_head_roundtrip() {
    if skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, d) = (2, 32);
    let mut qk = QuantKTurbo3::new(vec![1, kv_h, 0, d], 512);
    let c0 = rt_head_major_chunk(kv_h, 2, d, 0);
    let c1 = rt_head_major_chunk(kv_h, 1, d, 2);
    qk.append(
        &[],
        &[1, kv_h, 2, d],
        &rt_f32_array(&c0, &[1, kv_h, 2, d]),
        Device::Gpu,
        512,
    )
    .expect("append0");
    qk.append(
        &[],
        &[1, kv_h, 1, d],
        &rt_f32_array(&c1, &[1, kv_h, 1, d]),
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
        "turbo3 GPU kv_h=2 two-append max abs error {m} — expected q3 noise, not head scramble"
    );
}

// ── Parity: CPU == MSL (ignored, requires Metal) ─────────────────────────────

/// Parity test: CPU path vs MSL kernel — both must decode to within 1e-5.
///
/// `#[ignore]` — requires Metal GPU context. Run explicitly:
///   `cargo test -p rmlx-kv-quant quant_k_turbo3_cpu_msl_parity -- --ignored`
///
/// Skips silently when `RMLX_SKIP_GPU=1` (CI without Metal).
///
/// The GPU path quantizes via the `turbo_quantize_v3_gpu` MSL kernel; the
/// CPU path uses the Rust scalar `turbo_quantize_v`. Both pack to the same
/// 3-bit Lloyd-Max codebook so decoded output must match within float rounding.
#[test]
#[ignore = "GPU Metal context — run explicitly"]
fn quant_k_turbo3_cpu_msl_parity() {
    if skip_if_no_gpu_env() {
        return;
    }

    // SAFETY: `f32` is `Copy` with a fixed 4-byte little-endian layout.
    // `Array::from_bytes` copies immediately; the temporary bytes are not retained.
    #[allow(unsafe_code)]
    #[allow(
        clippy::items_after_statements,
        reason = "helper is local to this test; placement here is intentional for readability"
    )]
    fn make_f32_array(vals: &[f32], shape: &[i32]) -> Array {
        let bytes =
            unsafe { std::slice::from_raw_parts(vals.as_ptr().cast::<u8>(), vals.len() * 4) };
        Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
    }

    let head_dim = 128;
    let n_rows = 8;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    // CPU path
    let mut qk_cpu = QuantKTurbo3::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    cpu_append(&mut qk_cpu, &data, &new_shape);
    let cpu_out = qk_cpu.dequant().expect("cpu dequant must succeed");

    // GPU path: pass the real Array so the GPU kernel has data
    let k_arr = make_f32_array(&data, &new_shape);
    let mut qk_gpu = QuantKTurbo3::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    qk_gpu
        .append(&data, &new_shape, &k_arr, Device::Gpu, n_rows as i32)
        .expect("GPU append must succeed");
    let (gpu_out, gpu_arr_opt) = qk_gpu
        .dequantize_choice(Device::Gpu, Dtype::F32)
        .expect("GPU dequant must succeed");

    // GPU path returns an Array; convert to Vec<f32> via to_bytes().
    // The GPU Array from turbo_dequantize_v3_gpu is already materialized;
    // to_bytes() copies the raw f32 bytes out without a separate eval call.
    #[allow(unsafe_code)]
    let gpu_vec: Vec<f32> = if let Some(gpu_arr) = gpu_arr_opt {
        let bytes = gpu_arr.to_bytes().expect("GPU array to_bytes must succeed");
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n);
        // SAFETY: f32 and u8 have no validity requirements; we copy immediately.
        // `bytes` from `to_bytes()` is 4-byte aligned (MLX guarantees aligned
        // buffers); cast to `*const f32` is sound. clippy::cast_ptr_alignment
        // is suppressed here because it cannot reason about MLX's guarantees.
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            let ptr = bytes.as_ptr().cast::<f32>();
            out.extend_from_slice(std::slice::from_raw_parts(ptr, n));
        }
        out
    } else {
        gpu_out
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

    // The turbo3 codebook is hardwired identically in Rust and MSL; any
    // non-trivial deviation is a codec discrepancy.
    assert!(
        max_err < 1e-5,
        "CPU vs MSL max-abs-error = {max_err:.2e} exceeds 1e-5 (codec discrepancy)"
    );
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
/// concatenation back in `QuantKTurbo3::dequantize_choice` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_k_turbo3_two_block_decode_matches_one_block_at_b_gt_1() {
    for b in [1_usize, 2] {
        let (kv_h, head_dim) = (2_usize, 32_usize);
        let (n0, n1) = (2_usize, 3_usize);
        let max_seq = 512_i32;
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];
        let dummy =
            |n: usize| rmlx_mlx::zeros(&shape(n), Dtype::F32, Device::Cpu).expect("dummy array");
        let cpu_dequant = |st: &QuantKTurbo3| {
            st.dequantize_choice(Device::Cpu, Dtype::F32)
                .expect("cpu dequant")
                .0
        };

        let mut one = QuantKTurbo3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], max_seq);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
            &dummy(n0 + n1),
            Device::Cpu,
            max_seq,
        )
        .expect("single append");
        let oracle = cpu_dequant(&one);

        let mut two = QuantKTurbo3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], max_seq);
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
