use super::*;
use crate::c_api::{self, CApi};
use crate::Dtype;

/// Verify rope_dynamic (offset as 0-D i32 array) produces matching output
/// to rope (offset as captured i32) for a representative shape.
///
/// Foundational invariant for using rope_dynamic inside an mx.compile
/// closure: the dynamic-offset path MUST match the static-offset path at
/// every step — otherwise the model's positional embeddings drift.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rope_dynamic_matches_static -- --ignored --test-threads=1"]
fn rope_dynamic_matches_static() {
    // Shape [B=1, H=2, S=1, D=8] — single-token decode step.
    let b = 1_i32;
    let h = 2_i32;
    let s = 1_i32;
    let d = 8_i32;
    let n = (b * h * s * d) as usize;
    // Deterministic data.
    let data: Vec<f32> = (0..n).map(|i| (i as f32).mul_add(0.05, -0.4)).collect();
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    let x = Array::from_bytes(bytes, &[b, h, s, d], Dtype::F32).expect("from_bytes x");

    let offset_val: i32 = 17;
    let base = 10000.0_f32;
    let scale = 1.0_f32;

    // Static-offset reference.
    let y_static = rope(&x, d, false, base, scale, offset_val, Device::Gpu).expect("rope static");
    Array::eval(&y_static).expect("materialize static");
    let bytes_static = y_static.to_bytes().expect("to_bytes static");

    // Dynamic-offset path.
    let off_bytes = offset_val.to_le_bytes();
    let off_arr = Array::from_bytes(&off_bytes, &[], Dtype::I32).expect("from_bytes offset");
    let y_dyn =
        rope_dynamic(&x, d, false, base, scale, &off_arr, Device::Gpu).expect("rope_dynamic");
    Array::eval(&y_dyn).expect("materialize dynamic");
    let bytes_dyn = y_dyn.to_bytes().expect("to_bytes dynamic");

    assert_eq!(bytes_static.len(), bytes_dyn.len(), "byte-len mismatch");
    let sf: Vec<f32> = bytes_static
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let df: Vec<f32> = bytes_dyn
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    for (i, (sv, dv)) in sf.iter().zip(df.iter()).enumerate() {
        let diff = (sv - dv).abs();
        assert!(
            diff < 1e-5,
            "rope_dynamic vs rope mismatch at idx {i}: static={sv} dyn={dv} diff={diff}"
        );
    }
}

/// One query, two keys, head_dim 2, scale 1: the scores are [1, 0], so the
/// output is `softmax([1, 0]) @ V` = [1 + 2w, 2 + 2w] with w = 1 / (1 + e).
fn one_head_attention_inputs() -> Result<(Array, Array, Array)> {
    let q = Array::from_f32_slice(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let k = Array::from_f32_slice(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
    let v = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2])?;
    Ok((q, k, v))
}

fn one_head_attention_reference() -> Vec<f32> {
    let w = 1.0 / (1.0 + std::f32::consts::E);
    vec![2.0f32.mul_add(w, 1.0), 2.0f32.mul_add(w, 2.0)]
}

/// The CPU stream runs MLX's composite SDPA graph, so this needs no Metal.
/// A wrong argument list in the call does not compile, and `force_fused =
/// true` fails here: the CPU stream has no fused kernel to force.
///
/// On a pair whose `libmlxc.dylib` has another C API than the compiled one,
/// the call must be refused instead, with nothing passed to mlx-c.
#[test]
fn scaled_dot_product_attention_on_cpu_follows_the_c_api_verdict() {
    let (q, k, v) = one_head_attention_inputs().expect("attention inputs");
    let out = scaled_dot_product_attention(&q, &k, &v, 1.0, "", None, Device::Cpu);
    match c_api::verdict() {
        CApiVerdict::Match(_) => {
            let got: Vec<f32> = out
                .expect("sdpa on the CPU stream")
                .to_bytes()
                .expect("to_bytes")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let want = one_head_attention_reference();
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-6, "sdpa {got:?} != reference {want:?}");
            }
            assert_eq!(got.len(), want.len());
        }
        CApiVerdict::Mismatch { .. } => {
            let err = out.expect_err("a C API mismatch must refuse the call");
            assert!(err.to_string().contains("mlx-c C API mismatch"), "{err}");
        }
    }
}

/// A mismatch verdict stops the call before mlx-c sees it. On a matched pair
/// the mlx-c call itself would succeed, so an `Ok` here means the check is
/// gone or comes after the call.
#[test]
fn a_c_api_mismatch_refuses_sdpa_before_the_mlx_c_call() {
    let (q, k, v) = one_head_attention_inputs().expect("attention inputs");
    let mismatch = CApiVerdict::Mismatch {
        compiled: CApi::COMPILED,
        loaded: CApi::COMPILED.other(),
    };
    let err = sdpa_under(mismatch, &q, &k, &v, 1.0, "", None, Device::Cpu)
        .expect_err("a C API mismatch must refuse the call");
    assert!(err.to_string().contains("mlx-c C API mismatch"), "{err}");
}

/// Uniform values in `[-1, 1)` from a fixed sequence, as bf16 on the GPU.
fn uniform_bf16(state: &mut u64, shape: &[i32]) -> Array {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect();
    Array::from_f32_slice(&data, shape)
        .expect("host array")
        .astype(Dtype::Bf16, Device::Gpu)
        .expect("astype bf16")
}

fn f32_bits(a: &Array) -> Vec<u32> {
    let f32_array = a.astype(Dtype::F32, Device::Gpu).expect("astype f32");
    f32_array.eval().expect("materialise");
    f32_array
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The call the rule pads, against the same call handed to MLX as it is.
///
/// Only this crate can build the unpadded node: every other caller passes the
/// rule. The key rows are a multiple of 32, so the unpadded node is not in the
/// configuration that fails under the instrument, and this test reads the
/// same under the GPU runner's instrumented pass. The rule is on the query
/// rows alone and pads these calls.
///
/// The bound is bit equality, and it is the measured result on these two
/// shapes. MLX compiles the kernel for "query rows a multiple of 64" or not.
/// That the two differ only in how the last partial block of query rows is
/// loaded is a reading of the kernel, not checked in its source.
#[test]
#[ignore = "evaluates attention on the Metal GPU"]
fn a_padded_call_returns_the_rows_of_the_unpadded_node() {
    // (query rows, key rows, query heads, kv heads): the Qwen3.5 head layout,
    // and one KV head with the query rows as long as the keys allow.
    for (q_rows, k_rows, q_heads, kv_heads) in [(1025, 3072, 8, 2), (1296, 1312, 8, 1)] {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let q = uniform_bf16(&mut state, &[1, q_heads, q_rows, 256]);
        let k = uniform_bf16(&mut state, &[1, kv_heads, k_rows, 256]);
        let v = uniform_bf16(&mut state, &[1, kv_heads, k_rows, 256]);
        let first_query = k_rows - q_rows;
        let padded_rows = (q_rows + 63) / 64 * 64;
        let mask = causal_mask_bf16(q_rows, k_rows, first_query);
        let mask_at_padded_rows = causal_mask_bf16(padded_rows, k_rows, first_query);

        assert_eq!(
            padded_query_rows(&q, &v, "array", Device::Gpu),
            Some((q_rows, padded_rows)),
            "the rule must pad this call"
        );
        let unpadded = attention_node(&q, &k, &v, 0.5, "array", Some(&mask), Device::Gpu, None)
            .expect("unpadded node");
        let padded = sdpa_under(
            c_api::verdict(),
            &q,
            &k,
            &v,
            0.5,
            "array",
            Some(&mask),
            Device::Gpu,
        )
        .expect("padded call");
        let pre_padded = sdpa_under(
            c_api::verdict(),
            &q,
            &k,
            &v,
            0.5,
            "array",
            Some(&mask_at_padded_rows),
            Device::Gpu,
        )
        .expect("pre-padded call");
        assert_eq!(padded.shape(), unpadded.shape());
        assert_eq!(pre_padded.shape(), unpadded.shape());
        let (padded, pre_padded, unpadded) = (
            f32_bits(&padded),
            f32_bits(&pre_padded),
            f32_bits(&unpadded),
        );
        assert!(padded.iter().all(|x| f32::from_bits(*x).is_finite()));
        let first = padded.iter().zip(&unpadded).position(|(a, b)| a != b);
        assert_eq!(
            first, None,
            "{q_rows} query rows over {k_rows} keys: the padded call differs from the unpadded \
             node at flat index {first:?}"
        );
        let first = pre_padded.iter().zip(&unpadded).position(|(a, b)| a != b);
        assert_eq!(
            first, None,
            "{q_rows} query rows over {k_rows} keys: the call with a mask at the padded rows \
             differs from the unpadded node at flat index {first:?}"
        );
    }
}

/// Evaluate `a` now, so a later bracket does not count its allocation.
fn materialised(a: Array) -> Array {
    a.eval().expect("materialise");
    a
}

/// The boolean causal mask `[1, 1, rows, key_rows]` (true = attend) for query
/// rows at key positions `first_query_position + i`. Rows past the last key
/// are open.
fn causal_allowed(rows: i32, key_rows: i32, first_query_position: i32) -> Array {
    let q_pos = crate::arange(
        f64::from(first_query_position),
        f64::from(first_query_position + rows),
        1.0,
        Device::Gpu,
    )
    .expect("query positions")
    .reshape(&[1, 1, rows, 1], Device::Gpu)
    .expect("query column");
    let k_pos = crate::arange(0.0, f64::from(key_rows), 1.0, Device::Gpu)
        .expect("key positions")
        .reshape(&[1, 1, 1, key_rows], Device::Gpu)
        .expect("key row");
    materialised(crate::greater_equal(&q_pos, &k_pos, Device::Gpu).expect("allowed"))
}

/// [`causal_allowed`] as an additive bf16 mask.
fn causal_mask_bf16(rows: i32, key_rows: i32, first_query_position: i32) -> Array {
    let allowed = causal_allowed(rows, key_rows, first_query_position);
    let open = crate::scalar_f32(0.0)
        .astype(Dtype::Bf16, Device::Gpu)
        .expect("open");
    let blocked = crate::scalar_f32(-1e30)
        .astype(Dtype::Bf16, Device::Gpu)
        .expect("blocked");
    materialised(crate::where_cond(&allowed, &open, &blocked, Device::Gpu).expect("mask"))
}

/// MLX frees the inputs of a command buffer when it completes, which can be
/// after the evaluation that waited on it returned. Wait until the live byte
/// count has not moved for 100 ms (at most 5 s), so a bracket opens on a
/// settled count.
fn wait_for_buffers_to_be_released() {
    let step = std::time::Duration::from_millis(100);
    let mut last = crate::mlx_active_memory_bytes();
    for _ in 0..50 {
        std::thread::sleep(step);
        let now = crate::mlx_active_memory_bytes();
        if now == last {
            return;
        }
        last = now;
    }
    panic!("the live byte count did not settle in 5 s");
}

/// Bytes above the live bytes at open that one call needed, the output
/// evaluated inside the bracket. The output is returned so the caller can
/// compare it; it is evaluated before the bracket closes.
fn headroom_of_call(call: impl FnOnce() -> Array) -> (u64, Array) {
    wait_for_buffers_to_be_released();
    let bracket = crate::PeakBracket::open();
    let out = materialised(call());
    let reading = bracket.close();
    assert!(
        reading.measurable(),
        "the peak mark could not be reset, so there is no reading"
    );
    (reading.headroom_bytes(), out)
}

/// The mask copy of the row rule is gone when the caller hands the mask at the
/// padded rows. One Qwen3.8 layer (24 query heads, 4 KV heads, head dim 256,
/// 1766 query rows of a 2048-row chunk) at two key counts.
///
/// The bound is declared before the measurement: the call that copies the mask
/// needs at least 95 % of the copy's bytes (`1792 x key rows x 2`) more than
/// the call whose mask is already at 1792 rows, at both key counts; and the
/// headroom of the pre-padded call does not grow with the key rows (the two
/// counts differ by less than 5 % of the copy's growth). The two outputs are
/// bit-equal. Neither key count is a multiple of 32, so the unpadded node is
/// the configuration that fails under the shader-validation instrument: it is
/// compared in the padded-against-unpadded cell, at aligned key rows.
#[test]
#[ignore = "evaluates attention on the Metal GPU"]
fn a_padded_call_needs_no_mask_copy_when_the_mask_is_at_the_padded_rows() {
    const QUERY_ROWS: i32 = 1766;
    const PADDED_ROWS: i32 = 1792;
    let mut pre_padded_headroom = Vec::new();
    for key_rows in [16_102i32, 32_486] {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let q = materialised(uniform_bf16(&mut state, &[1, 24, QUERY_ROWS, 256]));
        let k = materialised(uniform_bf16(&mut state, &[1, 4, key_rows, 256]));
        let v = materialised(uniform_bf16(&mut state, &[1, 4, key_rows, 256]));
        let first_query = key_rows - QUERY_ROWS;
        let mask = causal_mask_bf16(QUERY_ROWS, key_rows, first_query);
        let mask_at_padded_rows = causal_mask_bf16(PADDED_ROWS, key_rows, first_query);
        let call = |mask: &Array| {
            sdpa_under(
                c_api::verdict(),
                &q,
                &k,
                &v,
                0.0625,
                "array",
                Some(mask),
                Device::Gpu,
            )
            .expect("padded call")
        };

        let (copy_headroom, copy_out) = headroom_of_call(|| call(&mask));
        let (pre_padded, pre_padded_out) = headroom_of_call(|| call(&mask_at_padded_rows));
        let mask_copy_bytes = (PADDED_ROWS as u64) * (key_rows as u64) * 2;
        assert!(
            copy_headroom.saturating_sub(pre_padded) >= mask_copy_bytes * 95 / 100,
            "{key_rows} keys: the copying call needed {copy_headroom} B and the pre-padded call \
             {pre_padded} B; the mask copy is {mask_copy_bytes} B"
        );
        println!(
            "headroom at {key_rows} keys: copying call {copy_headroom} B, pre-padded call \
             {pre_padded} B, difference {} B, mask copy {mask_copy_bytes} B",
            copy_headroom.saturating_sub(pre_padded)
        );
        pre_padded_headroom.push(pre_padded);

        assert_eq!(
            f32_bits(&copy_out),
            f32_bits(&pre_padded_out),
            "{key_rows} keys: the two calls differ"
        );
    }
    let growth = pre_padded_headroom[0].abs_diff(pre_padded_headroom[1]);
    let copy_growth = (PADDED_ROWS as u64) * (32_486 - 16_102) * 2;
    assert!(
        growth < copy_growth / 20,
        "the pre-padded call's headroom grew by {growth} B between 16102 and 32486 keys \
         ({pre_padded_headroom:?}); a mask copy would grow it by {copy_growth} B"
    );
}

/// The node's rows are the next multiple of 64 for exactly the calls the rule
/// pads, and the call's own rows for every other.
// gpu-test-gate: exempt
#[test]
fn the_node_rows_are_padded_for_exactly_the_calls_the_rule_pads() {
    let rows = |query_rows, head_dim, v_head_dim, mode, device| {
        attention_node_query_rows(query_rows, head_dim, v_head_dim, mode, device)
    };
    assert_eq!(rows(1766, 256, 256, "array", Device::Gpu), 1792);
    assert_eq!(rows(1025, 256, 256, "array", Device::Gpu), 1088);
    assert_eq!(rows(2047, 256, 256, "array", Device::Gpu), 2048);
    assert_eq!(rows(1024, 256, 256, "array", Device::Gpu), 1024);
    assert_eq!(rows(2048, 256, 256, "array", Device::Gpu), 2048);
    assert_eq!(rows(1023, 256, 256, "array", Device::Gpu), 1023);
    assert_eq!(rows(1, 256, 256, "array", Device::Gpu), 1);
    assert_eq!(rows(1766, 256, 256, "causal", Device::Gpu), 1766);
    assert_eq!(rows(1766, 256, 256, "", Device::Gpu), 1766);
    assert_eq!(rows(1766, 256, 256, "array", Device::Cpu), 1766);
    assert_eq!(rows(1766, 128, 256, "array", Device::Gpu), 1766);
    assert_eq!(rows(1766, 256, 128, "array", Device::Gpu), 1766);
}

/// A mask with one row more or fewer than the padded rows is refused, as is
/// any other row count: the wrapper pads only the call's own rows.
#[test]
#[ignore = "builds attention nodes on the Metal GPU stream"]
fn a_padded_call_refuses_a_mask_one_row_off_the_padded_rows() {
    let (q_rows, k_rows) = (1025, 3072);
    let q = crate::zeros(&[1, 8, q_rows, 256], Dtype::Bf16, Device::Gpu).expect("q");
    let kv = crate::zeros(&[1, 2, k_rows, 256], Dtype::Bf16, Device::Gpu).expect("kv");
    for rows in [1087, 1089] {
        let mask = causal_mask_bf16(rows, k_rows, k_rows - q_rows);
        let err = sdpa_under(
            c_api::verdict(),
            &q,
            &kv,
            &kv,
            0.5,
            "array",
            Some(&mask),
            Device::Gpu,
        )
        .expect_err("a mask that is neither the query rows nor the padded rows");
        assert!(err.to_string().contains("query axis"), "{rows} rows: {err}");
    }
}

/// A boolean mask already at the padded rows goes to MLX as it is and gives
/// the rows of the call with the mask at its own rows.
#[test]
#[ignore = "evaluates attention on the Metal GPU"]
fn a_padded_call_takes_a_boolean_mask_at_the_padded_rows_as_it_is() {
    let (q_rows, k_rows, padded_rows) = (1025, 3072, 1088);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let q = uniform_bf16(&mut state, &[1, 8, q_rows, 256]);
    let k = uniform_bf16(&mut state, &[1, 2, k_rows, 256]);
    let v = uniform_bf16(&mut state, &[1, 2, k_rows, 256]);
    let first_query = k_rows - q_rows;
    let call = |mask: &Array| {
        sdpa_under(
            c_api::verdict(),
            &q,
            &k,
            &v,
            0.5,
            "array",
            Some(mask),
            Device::Gpu,
        )
        .expect("call")
    };
    let own_rows = f32_bits(&call(&causal_allowed(q_rows, k_rows, first_query)));
    let padded = f32_bits(&call(&causal_allowed(padded_rows, k_rows, first_query)));
    assert_eq!(own_rows.len(), 8 * q_rows as usize * 256);
    assert!(own_rows == padded, "the two calls differ");
}
