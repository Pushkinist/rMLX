//! Unit tests for [`QuantIsoV3`] (includes GPU-mirror path).

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::quant_iso_v::{QuantIsoV3, ISO3_BITS, ISO3_GROUP_SIZE};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env, TEST_SEED};
use rmlx_mlx::{Array, Device, Dtype};

/// Provisioned window for the GPU-mirror tests. Larger than any chunk they
/// append, so the mirror grows by pages rather than clamping.
const ISO_TEST_MAX_SEQ: i32 = 4096;

/// Newly-constructed `QuantIsoV3` carries the requested init shape and bit
/// width; no blocks yet.
#[test]
fn quant_iso_v_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let q = QuantIsoV3::new(init_shape.clone());
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.bits, ISO3_BITS, "bits should be ISO3_BITS (3)");
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert_eq!(q.byte_size(), 0, "byte_size 0 with no blocks");
}

/// Roundtrip: encode → store in QuantIsoV3 → `dequant` → compare against the
/// raw `iso_decode_fast` reference. Equal element-by-element.
#[test]
fn quant_iso_v_roundtrip_dequant() {
    // Small fixture: B=1, kv_h=2, n_seq=4, head_dim=8 → 8 tokens of length 8.
    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 8;
    let n_rows = b * kv_h * n_seq; // 8 tokens
    let data = lcg_data(n_rows * head_dim, TEST_SEED);

    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qv = QuantIsoV3::new(vec![b as i32, kv_h as i32, 0_i32, head_dim as i32]);
    qv.append(&data, &new_shape).expect("append should succeed");

    assert_eq!(qv.blocks.len(), 1, "one append → one block");
    assert_eq!(qv.shape[2], n_seq as i32, "shape[2] advanced by n_seq");
    assert!(
        qv.byte_size() > 0,
        "byte_size should be non-zero after append"
    );

    let decoded = qv.dequant().expect("dequant should succeed");

    // Reference: call iso_decode_fast directly on the same codes.
    let (ref_codes, ref_scales, ref_quats, ref_norms) =
        iso_encode_fast(&data, head_dim, ISO3_GROUP_SIZE, ISO3_BITS).expect("encode reference");
    let reference = iso_decode_fast(
        &ref_codes,
        &ref_scales,
        &ref_quats,
        &ref_norms,
        head_dim,
        ISO3_GROUP_SIZE,
        ISO3_BITS,
    )
    .expect("decode reference");

    assert_eq!(
        decoded.len(),
        reference.len(),
        "QuantIsoV3::dequant length should match iso_decode_fast"
    );

    let mut max_abs_err = 0.0_f32;
    for (a, b) in decoded.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        if d > max_abs_err {
            max_abs_err = d;
        }
    }
    assert!(
        max_abs_err < 1e-3,
        "QuantIsoV3::dequant vs iso_decode_fast max_abs_err = {max_abs_err:.6} (>= 1e-3)"
    );
}

/// Multi-append with `kv_h > 1` must produce the same dequant output as a
/// single-shot append of the concatenated head-major buffer. This is the
/// head↔sequence layout invariant: the CPU blocks accumulate one chunk per
/// append, and a head-major-per-block layout transposes heads across appends
/// when the dequant reshapes head-major over the full sequence.
///
/// Fixture: per-(head, token) distinct values so any head transposition shows
/// up as a large error (>> quant noise).
#[test]
fn quant_iso_v_multi_append_matches_single_shot_gqa() {
    let b = 1_usize;
    let kv_h = 3_usize;
    let head_dim = 8_usize;
    let chunk_a = 2_usize;
    let chunk_b = 3_usize;
    let s_total = chunk_a + chunk_b;

    // Distinct, recognisable per-(head, token, dim) values. Encode the full
    // head-major `[B, kv_h, S, D]` buffer once as the reference.
    let val = |h: usize, s: usize, d: usize| -> f32 {
        (h as f32) * 100.0 + (s as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };
    let mut full = vec![0.0_f32; b * kv_h * s_total * head_dim];
    for h in 0..kv_h {
        for s in 0..s_total {
            for d in 0..head_dim {
                full[(h * s_total + s) * head_dim + d] = val(h, s, d);
            }
        }
    }

    // Reference: one append of the whole sequence.
    let mut qref = QuantIsoV3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32]);
    qref.append(
        &full,
        &[b as i32, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .expect("single-shot append");
    let reference = qref.dequant().expect("single-shot dequant");

    // Two appends: chunk A then chunk B, each a head-major `[B, kv_h, s, D]`.
    let extract = |s_lo: usize, s_hi: usize| -> Vec<f32> {
        let s = s_hi - s_lo;
        let mut out = vec![0.0_f32; b * kv_h * s * head_dim];
        for h in 0..kv_h {
            for si in 0..s {
                for d in 0..head_dim {
                    out[(h * s + si) * head_dim + d] = val(h, s_lo + si, d);
                }
            }
        }
        out
    };
    let mut qv = QuantIsoV3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32]);
    qv.append(
        &extract(0, chunk_a),
        &[b as i32, kv_h as i32, chunk_a as i32, head_dim as i32],
    )
    .expect("append chunk A");
    qv.append(
        &extract(chunk_a, s_total),
        &[b as i32, kv_h as i32, chunk_b as i32, head_dim as i32],
    )
    .expect("append chunk B");
    let multi = qv.dequant().expect("multi-append dequant");

    assert_eq!(multi.len(), reference.len(), "length parity");
    let mut max_abs = 0.0_f32;
    for (a, b) in multi.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    // A head transposition produces errors on the order of the value spread
    // (~100s). Quant noise on this fixture is well under 1.0. Use 1.0 as the
    // discriminating threshold.
    assert!(
        max_abs < 1.0,
        "multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — \
         head↔seq layout scramble"
    );
}

/// After `append` then `reset`, the storage reports seq = 0 and dequant returns
/// the zero-element prefix only.
#[test]
fn quant_iso_v_reset_clears_seq() {
    let head_dim = 8;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qv = QuantIsoV3::new(vec![1, 1, 0, head_dim as i32]);
    qv.append(&data, &new_shape).unwrap();
    assert_eq!(qv.shape[2], n_seq as i32);

    qv.reset();
    assert_eq!(qv.shape[2], 0, "shape[2] reset to 0");
    assert!(qv.blocks.is_empty(), "blocks cleared on reset");
}

/// After `append` of N tokens then `truncate_to(N/2)`, the dequant prefix
/// retains the first N/2 tokens (when appended one token per call).
#[test]
fn quant_iso_v_truncate_to_keeps_first_n() {
    let head_dim = 8;
    let n_seq_each = 1; // one token per append call so truncate boundaries align
    let total_tokens = 4;
    let data_full = lcg_data(total_tokens * head_dim, TEST_SEED);

    let mut qv = QuantIsoV3::new(vec![1, 1, 0, head_dim as i32]);
    for tok in 0..total_tokens {
        let row = &data_full[tok * head_dim..(tok + 1) * head_dim];
        let new_shape = [1_i32, 1, n_seq_each, head_dim as i32];
        qv.append(row, &new_shape).unwrap();
    }
    assert_eq!(qv.shape[2], total_tokens as i32);
    assert_eq!(qv.blocks.len(), total_tokens);

    let keep = (total_tokens / 2) as i32;
    qv.truncate_to(keep);
    assert_eq!(
        qv.shape[2], keep,
        "shape[2] should be `keep` after truncate"
    );
    assert_eq!(
        qv.blocks.len(),
        keep as usize,
        "block count should equal `keep` when one token per block"
    );

    // Decode the truncated prefix and compare against the reference for the first `keep` tokens.
    let decoded = qv.dequant().unwrap();
    // Reference: encode + decode the first `keep * head_dim` f32 of the original data.
    let prefix = &data_full[..(keep as usize) * head_dim];
    let (codes, scales, quats, norms) =
        iso_encode_fast(prefix, head_dim, ISO3_GROUP_SIZE, ISO3_BITS).unwrap();
    let reference = iso_decode_fast(
        &codes,
        &scales,
        &quats,
        &norms,
        head_dim,
        ISO3_GROUP_SIZE,
        ISO3_BITS,
    )
    .unwrap();

    assert_eq!(decoded.len(), reference.len());

    // Cosine match within tight tolerance.
    let stats = cosine_similarity_per_row(prefix, &decoded, head_dim);
    assert!(
        stats.min >= 0.99_f32,
        "truncated-prefix cosine min={:.6} below 0.99",
        stats.min
    );
    let _ = reference; // reference computed for shape parity check
}

// ── GPU-resident mirror tests ─────────────────────────────────────────────────
//
// All gated `#[ignore]` per Metal-context policy. Run explicitly via
//   cargo test -p rmlx-kv-quant --lib -- --ignored quant_iso_v --test-threads=1
//
// The GPU mirror default is OFF (hardcoded; no env-var opt-in). The mirror
// tests need it ON, so each test calls
// `force_gpu_resident_iso_on()` before any call into `append_gpu` —
// `gpu_resident_iso_enabled()` latches the value on first read via OnceLock,
// so the very first test to set it wins for the rest of the test-binary
// lifetime. The `--test-threads=1` invocation is required so there is no
// concurrent reader/writer of the AtomicBool + OnceLock pair.

/// Enable the GPU-resident ISO mirror for the rest of this test binary.
///
/// Calls `crate::set_gpu_resident_iso_for_test(true)` so the OnceLock in
/// `gpu_resident_iso_enabled()` latches `true` on first read. No-op if already
/// latched ON (OnceLock is write-once after first read). Requires
/// `--test-threads=1` — concurrent readers of the OnceLock must not race with
/// this store.
///
/// Any future gate-OFF + GPU test must run in its own test binary
/// (integration test in `tests/`) because the per-process OnceLock cannot
/// be reset once latched.
fn force_gpu_resident_iso_on() {
    crate::set_gpu_resident_iso_for_test(true);
}

/// Build a 4-D f32 `Array` from a row-major slice and shape `[B, kv_h, S, D]`.
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from Array::from_bytes is the desired test failure mode"
)]
#[allow(
    unsafe_code,
    reason = "test helper: reinterpret f32 slice as bytes for Array::from_bytes; \
              slice lifetime is tied to caller-owned `data`, no aliasing"
)]
fn make_f32_array_4d(data: &[f32], shape: &[i32; 4]) -> Array {
    // SAFETY: f32 has no padding, alignment 4; reinterpreting as u8 is
    // well-defined. The borrowed byte slice is copied immediately by
    // Array::from_bytes.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array_4d")
}

/// After a GPU encode, the per-struct GPU mirror is populated and
/// `gpu_offset` matches the cumulative token count.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored quant_iso_v --test-threads=1"]
fn iso_v3_gpu_mirror_populated_on_encode() {
    if skip_if_no_gpu_env() {
        return;
    }
    force_gpu_resident_iso_on();
    let b = 1;
    let kv_h = 2;
    let head_dim = 8;
    // Two chunks of 3 and 5 tokens — total 8.
    let chunk_sizes = [3_usize, 5];

    let mut qv = QuantIsoV3::new(vec![b, kv_h, 0_i32, head_dim]);
    let mut cum = 0_i32;
    for &s in &chunk_sizes {
        let total = (b as usize) * (kv_h as usize) * s * (head_dim as usize);
        let data = lcg_data(total, TEST_SEED.wrapping_add(s as u64));
        let arr = make_f32_array_4d(&data, &[b, kv_h, s as i32, head_dim]);
        qv.append_gpu(
            &arr,
            &[b, kv_h, s as i32, head_dim],
            ISO_TEST_MAX_SEQ,
            Device::Gpu,
        )
        .expect("append_gpu");
        cum += s as i32;
        assert_eq!(
            qv.gpu_offset, cum,
            "gpu_offset must advance by chunk_size each append (cum={cum})"
        );
        assert_eq!(qv.shape[2], cum, "shape[2] must equal gpu_offset");
    }
    assert!(
        qv.gpu_codes_buf.is_some(),
        "gpu_codes_buf must be Some after first GPU append"
    );
    assert!(qv.gpu_scales_buf.is_some(), "gpu_scales_buf must be Some");
    assert!(qv.gpu_norms_buf.is_some(), "gpu_norms_buf must be Some");
    // CPU blocks must also be populated (SSD spill compat).
    assert_eq!(
        qv.blocks.len(),
        chunk_sizes.len(),
        "CPU blocks must mirror per-chunk appends for SSD spill"
    );
}

/// `dequant_gpu` with a populated mirror must produce the same f32
/// output as the CPU-staged path (within codec tolerance).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored quant_iso_v --test-threads=1"]
fn iso_v3_dequant_gpu_uses_mirror_when_populated() {
    if skip_if_no_gpu_env() {
        return;
    }
    force_gpu_resident_iso_on();
    let b = 1;
    let kv_h = 2;
    let head_dim = 8;
    let s = 4;
    let total = (b as usize) * (kv_h as usize) * (s as usize) * (head_dim as usize);
    let data = lcg_data(total, TEST_SEED);
    let arr = make_f32_array_4d(&data, &[b, kv_h, s, head_dim]);

    let mut qv = QuantIsoV3::new(vec![b, kv_h, 0_i32, head_dim]);
    qv.append_gpu(&arr, &[b, kv_h, s, head_dim], ISO_TEST_MAX_SEQ, Device::Gpu)
        .expect("append_gpu");
    assert!(qv.gpu_codes_buf.is_some(), "mirror must be populated");

    // Fast path — mirror is consumed directly by dequant_gpu.
    let out_arr = qv.dequant_gpu(Device::Gpu).expect("dequant_gpu fast");
    out_arr.eval().expect("eval");
    let fast_bytes = out_arr.to_bytes().expect("to_bytes");
    let fast: Vec<f32> = fast_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect();

    // Slow path — clone CPU blocks into a fresh QuantIsoV3 with no mirror.
    let qv_no_mirror = QuantIsoV3::from_cpu_blocks(qv.blocks.clone(), qv.shape.clone());
    assert!(
        qv_no_mirror.gpu_codes_buf.is_none(),
        "from_cpu_blocks must leave mirror unallocated"
    );
    let slow_arr = qv_no_mirror
        .dequant_gpu(Device::Gpu)
        .expect("dequant_gpu fallback");
    slow_arr.eval().expect("eval");
    let slow_bytes = slow_arr.to_bytes().expect("to_bytes");
    let slow: Vec<f32> = slow_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect();

    assert_eq!(fast.len(), slow.len(), "fast/slow length parity");
    let mut max_abs = 0.0_f32;
    for (a, b) in fast.iter().zip(slow.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    // Both paths are GPU-side; should be bit-identical or within Metal f32
    // rounding noise.
    assert!(
        max_abs < 1e-3_f32,
        "fast vs fallback dequant_gpu max_abs_err = {max_abs:.6} (>= 1e-3)"
    );
}

/// When the mirror is cleared (post-SSD-hydrate behaviour),
/// `dequant_gpu` must transparently fall back to the CPU-staged
/// upload path and still produce a valid dequantised Array.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored quant_iso_v --test-threads=1"]
fn iso_v3_dequant_gpu_falls_back_to_cpu_path_when_mirror_missing() {
    if skip_if_no_gpu_env() {
        return;
    }
    force_gpu_resident_iso_on();
    let b = 1;
    let kv_h = 1;
    let head_dim = 8;
    let s = 4;
    let total = (b as usize) * (kv_h as usize) * (s as usize) * (head_dim as usize);
    let data = lcg_data(total, TEST_SEED);
    let arr = make_f32_array_4d(&data, &[b, kv_h, s, head_dim]);

    let mut qv = QuantIsoV3::new(vec![b, kv_h, 0_i32, head_dim]);
    qv.append_gpu(&arr, &[b, kv_h, s, head_dim], ISO_TEST_MAX_SEQ, Device::Gpu)
        .expect("append_gpu");

    // Simulate post-SSD-hydrate state: mirror dropped, CPU blocks retained.
    qv.gpu_codes_buf = None;
    qv.gpu_scales_buf = None;
    qv.gpu_norms_buf = None;
    qv.gpu_offset = 0;
    qv.gpu_capacity = 0;

    // Should still dequant cleanly via the CPU-block path.
    let out = qv
        .dequant_gpu(Device::Gpu)
        .expect("CPU-staged fallback dequant_gpu");
    assert_eq!(
        out.shape(),
        qv.shape,
        "fallback dequant_gpu shape must match storage shape"
    );
}

/// `reset` drops the GPU mirror so the next append re-allocates from scratch.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored quant_iso_v --test-threads=1"]
fn iso_v3_reset_clears_gpu_mirror() {
    if skip_if_no_gpu_env() {
        return;
    }
    force_gpu_resident_iso_on();
    let b = 1;
    let kv_h = 1;
    let head_dim = 8;
    let s = 4;
    let total = (b as usize) * (kv_h as usize) * (s as usize) * (head_dim as usize);
    let data = lcg_data(total, TEST_SEED);
    let arr = make_f32_array_4d(&data, &[b, kv_h, s, head_dim]);

    let mut qv = QuantIsoV3::new(vec![b, kv_h, 0_i32, head_dim]);
    qv.append_gpu(&arr, &[b, kv_h, s, head_dim], ISO_TEST_MAX_SEQ, Device::Gpu)
        .expect("append_gpu");
    assert!(qv.gpu_codes_buf.is_some());

    qv.reset();
    assert!(qv.gpu_codes_buf.is_none(), "reset must drop mirror codes");
    assert!(qv.gpu_scales_buf.is_none(), "reset must drop mirror scales");
    assert!(qv.gpu_norms_buf.is_none(), "reset must drop mirror norms");
    assert_eq!(qv.gpu_offset, 0, "reset must zero gpu_offset");
    assert_eq!(qv.gpu_capacity, 0, "reset must zero gpu_capacity");
    assert!(qv.blocks.is_empty(), "reset must clear CPU blocks");
}

/// GPU multi-append with `kv_h > 1` must match a single-shot GPU append of the
/// concatenated head-major buffer. Exercises the GPU-mirror layout invariant:
/// `append_gpu` reorders each chunk seq-major before the encode kernel, and
/// `dequant_gpu` reorders back — a head-major store + head-major reshape would
/// transpose heads across the two chunks. CPU tests cannot catch the MSL
/// raw-linear-index stride footgun, so this runs on the real GPU.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored quant_iso_v --test-threads=1"]
fn iso_v3_gpu_multi_append_matches_single_shot_gqa() {
    if skip_if_no_gpu_env() {
        return;
    }
    force_gpu_resident_iso_on();
    let b = 1_i32;
    let kv_h = 3_i32;
    let head_dim = 8_i32;
    let chunk_a = 2_i32;
    let chunk_b = 3_i32;
    let s_total = chunk_a + chunk_b;

    let val = |h: i32, s: i32, d: i32| -> f32 {
        (h as f32) * 100.0 + (s as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };
    let build = |s_lo: i32, s_hi: i32| -> Vec<f32> {
        let s = s_hi - s_lo;
        let mut out = vec![0.0_f32; (b * kv_h * s * head_dim) as usize];
        for h in 0..kv_h {
            for si in 0..s {
                for d in 0..head_dim {
                    let idx = (((h * s) + si) * head_dim + d) as usize;
                    out[idx] = val(h, s_lo + si, d);
                }
            }
        }
        out
    };

    // Reference: one GPU append of the full sequence.
    let full = build(0, s_total);
    let full_arr = make_f32_array_4d(&full, &[b, kv_h, s_total, head_dim]);
    let mut qref = QuantIsoV3::new(vec![b, kv_h, 0, head_dim]);
    qref.append_gpu(
        &full_arr,
        &[b, kv_h, s_total, head_dim],
        ISO_TEST_MAX_SEQ,
        Device::Gpu,
    )
    .expect("single-shot append_gpu");
    let ref_arr = qref
        .dequant_gpu(Device::Gpu)
        .expect("single-shot dequant_gpu");
    ref_arr.eval().expect("eval");
    let reference: Vec<f32> = ref_arr
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().expect("chunk len 4")))
        .collect();

    // Two GPU appends.
    let arr_a = make_f32_array_4d(&build(0, chunk_a), &[b, kv_h, chunk_a, head_dim]);
    let arr_b = make_f32_array_4d(&build(chunk_a, s_total), &[b, kv_h, chunk_b, head_dim]);
    let mut qv = QuantIsoV3::new(vec![b, kv_h, 0, head_dim]);
    qv.append_gpu(
        &arr_a,
        &[b, kv_h, chunk_a, head_dim],
        ISO_TEST_MAX_SEQ,
        Device::Gpu,
    )
    .expect("append_gpu A");
    qv.append_gpu(
        &arr_b,
        &[b, kv_h, chunk_b, head_dim],
        ISO_TEST_MAX_SEQ,
        Device::Gpu,
    )
    .expect("append_gpu B");
    let multi_arr = qv.dequant_gpu(Device::Gpu).expect("multi dequant_gpu");
    multi_arr.eval().expect("eval");
    let multi: Vec<f32> = multi_arr
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().expect("chunk len 4")))
        .collect();

    assert_eq!(multi.len(), reference.len(), "length parity");
    let mut max_abs = 0.0_f32;
    for (a, b) in multi.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    // A head transposition produces errors ~100s; quant noise on this fixture
    // is well under 1.0.
    assert!(
        max_abs < 1.0,
        "GPU multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — \
         head↔seq layout scramble"
    );

    // kv_h=1 control: layout reorder is a no-op, multi-append still matches.
    let head_dim1 = 8_i32;
    let c1 = build_single_head(b, head_dim1, 0, chunk_a, &val);
    let c2 = build_single_head(b, head_dim1, chunk_a, s_total, &val);
    let arr_c1 = make_f32_array_4d(&c1, &[b, 1, chunk_a, head_dim1]);
    let arr_c2 = make_f32_array_4d(&c2, &[b, 1, chunk_b, head_dim1]);
    let mut qctrl = QuantIsoV3::new(vec![b, 1, 0, head_dim1]);
    qctrl
        .append_gpu(
            &arr_c1,
            &[b, 1, chunk_a, head_dim1],
            ISO_TEST_MAX_SEQ,
            Device::Gpu,
        )
        .expect("ctrl append A");
    qctrl
        .append_gpu(
            &arr_c2,
            &[b, 1, chunk_b, head_dim1],
            ISO_TEST_MAX_SEQ,
            Device::Gpu,
        )
        .expect("ctrl append B");
    let ctrl_arr = qctrl.dequant_gpu(Device::Gpu).expect("ctrl dequant");
    ctrl_arr.eval().expect("eval");
    assert_eq!(
        ctrl_arr.shape(),
        vec![b, 1, s_total, head_dim1],
        "kv_h=1 control shape"
    );
}

/// Helper for the kv_h=1 control case: build a `[B, 1, s, D]` head-major chunk.
fn build_single_head(
    b: i32,
    head_dim: i32,
    s_lo: i32,
    s_hi: i32,
    val: &dyn Fn(i32, i32, i32) -> f32,
) -> Vec<f32> {
    let s = s_hi - s_lo;
    let mut out = vec![0.0_f32; (b * s * head_dim) as usize];
    for si in 0..s {
        for d in 0..head_dim {
            out[(si * head_dim + d) as usize] = val(0, s_lo + si, d);
        }
    }
    out
}

/// SSD spill→hydrate round-trip preserves the CPU blocks layout.
/// After hydrate the mirror is missing, so `dequant_gpu` falls back to the
/// CPU-staged path; the resulting output must be bit-identical (within codec
/// tolerance) to a pre-spill `dequant_gpu` on the same QuantIsoV3.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored quant_iso_v --test-threads=1"]
fn iso_v3_ssd_roundtrip_preserves_dequant_output() {
    if skip_if_no_gpu_env() {
        return;
    }
    force_gpu_resident_iso_on();
    let b = 1;
    let kv_h = 1;
    let head_dim = 8;
    let s = 4;
    let total = (b as usize) * (kv_h as usize) * (s as usize) * (head_dim as usize);
    let data = lcg_data(total, TEST_SEED);
    let arr = make_f32_array_4d(&data, &[b, kv_h, s, head_dim]);

    let mut qv = QuantIsoV3::new(vec![b, kv_h, 0_i32, head_dim]);
    qv.append_gpu(&arr, &[b, kv_h, s, head_dim], ISO_TEST_MAX_SEQ, Device::Gpu)
        .expect("append_gpu");

    // Pre-spill dequant via the GPU fast path.
    let pre_arr = qv.dequant_gpu(Device::Gpu).expect("pre dequant_gpu");
    pre_arr.eval().expect("eval");
    let pre: Vec<f32> = pre_arr
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect();

    // Simulate spill→hydrate: reconstruct from the CPU blocks only.
    let hydrated_blocks = qv.blocks.clone();
    let hydrated_shape = qv.shape.clone();
    let hydrated = QuantIsoV3::from_cpu_blocks(hydrated_blocks, hydrated_shape);
    assert!(
        hydrated.gpu_codes_buf.is_none(),
        "hydrate must leave mirror unallocated"
    );

    let post_arr = hydrated
        .dequant_gpu(Device::Gpu)
        .expect("post dequant_gpu (CPU-staged fallback)");
    post_arr.eval().expect("eval");
    let post: Vec<f32> = post_arr
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect();

    assert_eq!(pre.len(), post.len(), "pre/post dequant length parity");
    let mut max_abs = 0.0_f32;
    for (a, b) in pre.iter().zip(post.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    // Both paths route through the same MSL decode kernel; the only
    // difference is the source buffer location. Should be bit-identical.
    assert!(
        max_abs < 1e-5_f32,
        "spill→hydrate dequant divergence max_abs = {max_abs:.6} (>= 1e-5)"
    );
}

/// Falsifies #284: `quant_iso_v_truncate_to_keeps_first_n` above only runs at
/// `kv_h == 1`, where each block's `n_tokens` already equals its sequence
/// length (rows == seq). At `kv_h > 1`, `n_tokens` is inflated by `kv_h` and
/// `truncate_to(n)` must convert `n` to row units before comparing, or it
/// keeps `floor(n / kv_h)` blocks instead of `n`.
///
/// Builds one block per token (CPU-only, no GPU ring ever touched), truncates
/// mid-sequence at a block boundary, and requires the result to exactly match
/// a reference store built from only the first `keep_tokens`.
///
/// Mutation check: reverting `truncate_to` to compare
/// `acc + blk.n_tokens <= n as usize` (raw, not row-scaled) makes the
/// `kv_h > 1` case RED — `blocks.len()` drops and `dequant()` returns `Err`.
#[test]
fn quant_iso_v_truncate_to_kv_h_gt_1_keeps_exact_prefix() {
    let head_dim = 8_usize;
    let total_tokens = 4_usize;
    let keep_tokens = 2_usize;
    let val = |h: usize, tok: usize, d: usize| {
        (h as f32) * 100.0 + (tok as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };

    for kv_h in [1_usize, 4_usize] {
        let token_data = |tok: usize| -> Vec<f32> {
            let mut out = vec![0.0_f32; kv_h * head_dim];
            for h in 0..kv_h {
                for d in 0..head_dim {
                    out[h * head_dim + d] = val(h, tok, d);
                }
            }
            out
        };
        let new_shape = [1_i32, kv_h as i32, 1, head_dim as i32];

        let mut store = QuantIsoV3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32]);
        for tok in 0..total_tokens {
            store.append(&token_data(tok), &new_shape).unwrap();
        }
        assert_eq!(
            store.blocks.len(),
            total_tokens,
            "one block per token append (kv_h={kv_h})"
        );

        store.truncate_to(keep_tokens as i32);

        assert_eq!(
            store.shape[2], keep_tokens as i32,
            "shape[2] must equal keep_tokens (kv_h={kv_h})"
        );
        assert_eq!(
            store.blocks.len(),
            keep_tokens,
            "truncate_to must keep exactly keep_tokens blocks, not floor(keep_tokens / kv_h) (kv_h={kv_h})"
        );
        let kept_rows: usize = store.blocks.iter().map(|blk| blk.n_tokens).sum();
        assert_eq!(
            kept_rows,
            keep_tokens * kv_h,
            "kept rows must equal keep_tokens * b * kv_h (kv_h={kv_h})"
        );

        let decoded = store
            .dequant()
            .expect("dequant must succeed after truncate at kv_h>1 (#284)");

        let mut reference = QuantIsoV3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32]);
        for tok in 0..keep_tokens {
            reference.append(&token_data(tok), &new_shape).unwrap();
        }
        let ref_decoded = reference.dequant().unwrap();

        assert_eq!(
            decoded, ref_decoded,
            "truncated store must exactly match a store built from only the \
             first keep_tokens (kv_h={kv_h})"
        );
    }
}

/// The mid-block split covers the iso block layout too — a third per-row
/// buffer (`quaternions`) at yet another stride.
///
/// Same defect class as the rotor stores: a truncation target that lands inside
/// an append block used to drop the whole block, leaving `blocks` short of
/// `shape[2]` and `dequant()` aborting the request.
///
/// The oracle is a reference store built from only the retained tokens; it
/// shares no arithmetic with the truncation logic, which never reads a payload
/// value.
///
/// Mutation check: restore the whole-block drop and `dequant()` returns
/// `Err("iso V store: CPU blocks cover ...")`, so the `expect` goes RED.
#[test]
fn quant_iso_v3_truncate_mid_block_splits_instead_of_dropping() {
    let head_dim = 16_usize; // n_groups = 4, exact
    let kv_h = 2_usize;
    let chunk = |first_tok: usize, n_tok: usize| -> Vec<f32> {
        let mut out = vec![0.0_f32; kv_h * n_tok * head_dim];
        for h in 0..kv_h {
            for t in 0..n_tok {
                for d in 0..head_dim {
                    out[(h * n_tok + t) * head_dim + d] =
                        (h as f32) * 100.0 + (first_tok + t) as f32 * 10.0 + (d as f32) * 0.25;
                }
            }
        }
        out
    };

    let mut store = QuantIsoV3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32]);
    store
        .append(&chunk(0, 3), &[1, kv_h as i32, 3, head_dim as i32])
        .expect("first append");
    store
        .append(&chunk(3, 4), &[1, kv_h as i32, 4, head_dim as i32])
        .expect("second append");
    assert_eq!(store.shape[2], 7);

    store.truncate_to(5);

    let decoded = store
        .dequant()
        .expect("dequant must succeed after a mid-block truncate");

    let kept_rows: usize = store.blocks.iter().map(|blk| blk.n_tokens).sum();
    assert_eq!(kept_rows, 5 * kv_h, "blocks must cover shape[2] exactly");
    let quat_rows: usize = store
        .blocks
        .iter()
        .map(|blk| blk.quaternions.len() / (head_dim / ISO3_GROUP_SIZE * 4))
        .sum();
    assert_eq!(
        quat_rows,
        5 * kv_h,
        "the per-group quaternion buffer must be cut to the same row count"
    );

    let mut reference = QuantIsoV3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32]);
    reference
        .append(&chunk(0, 3), &[1, kv_h as i32, 3, head_dim as i32])
        .expect("first append");
    reference
        .append(&chunk(3, 2), &[1, kv_h as i32, 2, head_dim as i32])
        .expect("retained-prefix append");
    let ref_decoded = reference.dequant().expect("reference dequant");

    assert_eq!(
        decoded, ref_decoded,
        "the split block must reconstruct the retained prefix exactly"
    );
}
