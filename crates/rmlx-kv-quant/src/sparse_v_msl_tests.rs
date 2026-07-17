use super::*;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_u32_array(data: &[u32], shape: &[i32]) -> Array {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::U32).expect("make_u32_array")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Reference CPU: naive weighted sum without sparsity.
/// Affine dequant: val = scale * ((float)raw - midpoint) + bias.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn ref_weighted_sum(
    probs: &[f32],
    codes: &[u32],
    scales: &[f32],
    biases: &[f32],
    t_seq: usize,
    head_dim: usize,
    group_size: usize,
    bits: usize,
) -> Vec<f32> {
    let el_per_int = 32 / bits;
    let codes_d = head_dim / el_per_int;
    let scales_d = head_dim / group_size;
    let mask = (1u32 << bits) - 1;
    let midpoint = (1u32 << (bits - 1)) as f32;

    let mut out = vec![0.0f32; head_dim];
    for t in 0..t_seq {
        let p = probs[t];
        if p == 0.0 {
            continue;
        }
        for d in 0..head_dim {
            let d_word = d / el_per_int;
            let d_shift = (d % el_per_int) * bits;
            let raw = (codes[t * codes_d + d_word] >> d_shift) & mask;
            let code_float = raw as f32 - midpoint;
            let scale = scales[t * scales_d + d / group_size];
            let bias = biases[t * scales_d + d / group_size];
            out[d] += p * scale.mul_add(code_float, bias);
        }
    }
    out
}

/// Test: B=1, n_kv_heads=2, n_repeats=1, T_seq=128, head_dim=128, bits=8.
///
/// All codes = 128u (=> code_float = 0), scale=1, bias=0.5 => val=0.5.
/// Two sparse token positions per head: prob=0.5 and prob=0.5.
/// All other probs = 0 (skipped). Expected acc = 0.5 * 0.5 + 0.5 * 0.5 = 0.5.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test sparse_v_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn test_sparse_v_basic_8bit() {
    let device = Device::Gpu;
    let b = 1i32;
    let n_kv_heads = 2i32;
    let n_repeats = 1i32;
    let t_seq = 128i32;
    let head_dim = 128i32;
    let group_size = 128i32; // one group per token row
    let bits = 8i32;
    let el_per_int = 32 / bits; // = 4
    let codes_d = head_dim / el_per_int; // = 32
    let scales_d = head_dim / group_size; // = 1

    // code=128 unsigned => code_float = 128 - 128 = 0. scale=1, bias=0.5 => val=0.5.
    let code_word = 0x8080_8080u32;
    let n_codes = (b * n_kv_heads * t_seq * codes_d) as usize;
    let n_scales = (b * n_kv_heads * t_seq * scales_d) as usize;
    let n_probs = (b * n_kv_heads * n_repeats * t_seq) as usize;

    let codes_data: Vec<u32> = vec![code_word; n_codes];
    let scales_data: Vec<f32> = vec![1.0f32; n_scales];
    let biases_data: Vec<f32> = vec![0.5f32; n_scales];
    // Two active tokens per head: indices 0 and 64, each prob=0.5.
    let mut probs_data: Vec<f32> = vec![0.0f32; n_probs];
    for kv_h in 0..n_kv_heads as usize {
        probs_data[kv_h * t_seq as usize] = 0.5;
        probs_data[kv_h * t_seq as usize + 64] = 0.5;
    }

    let probs_arr = make_f32_array(&probs_data, &[b, n_kv_heads, n_repeats, 1, t_seq]);
    let codes_arr = make_u32_array(&codes_data, &[b, n_kv_heads, t_seq, codes_d]);
    let scales_arr = make_f32_array(&scales_data, &[b, n_kv_heads, t_seq, scales_d]);
    let biases_arr = make_f32_array(&biases_data, &[b, n_kv_heads, t_seq, scales_d]);

    let out = sparse_v_weighted_sum(
        &probs_arr,
        &codes_arr,
        &scales_arr,
        &biases_arr,
        b,
        n_kv_heads,
        n_repeats,
        t_seq,
        head_dim,
        group_size,
        bits,
        Dtype::F32,
        device,
    )
    .expect("sparse_v_weighted_sum");

    let result = to_f32_vec(&out);
    assert_eq!(
        result.len(),
        (b * n_kv_heads * n_repeats * head_dim) as usize
    );
    // acc = 0.5 * 0.5 + 0.5 * 0.5 = 0.5 for every dim.
    let expected_val = 0.5f32;
    for &v in &result {
        assert!(
            (v - expected_val).abs() < 1e-4,
            "expected {expected_val:.6}, got {v:.6}"
        );
    }
}

/// Test: 4-bit, compare GPU kernel against reference CPU for 2 KV heads.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test sparse_v_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn test_sparse_v_4bit_matches_reference() {
    let device = Device::Gpu;
    let b = 1i32;
    let n_kv_heads = 2i32;
    let n_repeats = 1i32;
    let t_seq = 8i32;
    let head_dim = 32i32;
    let group_size = 32i32; // one group per token row
    let bits = 4i32;
    let el_per_int = 8i32; // 32/4
    let codes_d = head_dim / el_per_int; // 4
    let scales_d = head_dim / group_size; // 1

    // Deterministic pseudo-random data via LCG.
    let mut state = 0xdeadbeef_u64;
    let mut next = move || -> f32 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f32) / (u32::MAX as f32)
    };

    let n_codes = (b * n_kv_heads * t_seq * codes_d) as usize;
    let n_scales = (b * n_kv_heads * t_seq * scales_d) as usize;
    let n_probs = (b * n_kv_heads * n_repeats * t_seq) as usize;

    let codes_data: Vec<u32> = (0..n_codes)
        .map(|_| (next() * u32::MAX as f32) as u32)
        .collect();
    let scales_data: Vec<f32> = (0..n_scales).map(|_| next() * 2.0 - 1.0).collect();
    let biases_data: Vec<f32> = (0..n_scales).map(|_| (next() - 0.5) * 0.1).collect();
    // Sparse probs: 80 % zeros.
    let probs_data: Vec<f32> = (0..n_probs)
        .map(|i| if i % 5 == 0 { next() } else { 0.0 })
        .collect();

    let probs_arr = make_f32_array(&probs_data, &[b, n_kv_heads, n_repeats, 1, t_seq]);
    let codes_arr = make_u32_array(&codes_data, &[b, n_kv_heads, t_seq, codes_d]);
    let scales_arr = make_f32_array(&scales_data, &[b, n_kv_heads, t_seq, scales_d]);
    let biases_arr = make_f32_array(&biases_data, &[b, n_kv_heads, t_seq, scales_d]);

    let out = sparse_v_weighted_sum(
        &probs_arr,
        &codes_arr,
        &scales_arr,
        &biases_arr,
        b,
        n_kv_heads,
        n_repeats,
        t_seq,
        head_dim,
        group_size,
        bits,
        Dtype::F32,
        device,
    )
    .expect("sparse_v_weighted_sum 4bit");

    let gpu_out = to_f32_vec(&out);

    // Compare per KV head.
    for kv_h in 0..n_kv_heads as usize {
        let p_off = kv_h * t_seq as usize;
        let c_off = kv_h * (t_seq * codes_d) as usize;
        let s_off = kv_h * (t_seq * scales_d) as usize;
        let ref_out = ref_weighted_sum(
            &probs_data[p_off..p_off + t_seq as usize],
            &codes_data[c_off..c_off + (t_seq * codes_d) as usize],
            &scales_data[s_off..s_off + (t_seq * scales_d) as usize],
            &biases_data[s_off..s_off + (t_seq * scales_d) as usize],
            t_seq as usize,
            head_dim as usize,
            group_size as usize,
            bits as usize,
        );
        let g_off = kv_h * head_dim as usize;
        for d in 0..head_dim as usize {
            let g = gpu_out[g_off + d];
            let r = ref_out[d];
            assert!(
                (g - r).abs() < 1e-4,
                "kv_h={kv_h} d={d}: gpu={g:.6} ref={r:.6}"
            );
        }
    }
}

/// Probe header snapshots must equal what the builders emit.
///
/// `make check-metal-compiles` prepends these snapshots to the kernel bodies.
/// A builder that changes a constant's value, or drops one, without the
/// snapshot being refreshed leaves the probe compiling text production no
/// longer emits — the gate would keep passing while checking the wrong thing.
/// Equality here turns that drift into a hard failure.
#[test]
fn hdr_probe_snapshot_matches_builder() {
    assert_eq!(
        build_kernel_header(kernel_eps()),
        include_str!("metal/probes/sparse_v.hdr.metal"),
        "stale snapshot: refresh metal/probes/sparse_v.hdr.metal"
    );
}
