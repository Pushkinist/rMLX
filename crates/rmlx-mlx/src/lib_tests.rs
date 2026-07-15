use super::*;

/// Reinterpret a &[f32] as &[u8] (no copy, no extra crates).
fn f32_as_bytes(s: &[f32]) -> &[u8] {
    // SAFETY: f32 is 4 bytes, both types have defined representations, and
    // the byte slice has the same lifetime as the input.
    unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), s.len() * 4) }
}

/// Convert little-endian f32 bytes to Vec<f32>.
fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    assert!(b.len().is_multiple_of(4));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── add ──────────────────────────────────────────────────────────────────

#[test]
fn add_two_f32_arrays() {
    let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let bytes = f32_as_bytes(&input);
    let shape = [2i32, 2];

    let a = Array::from_bytes(bytes, &shape, Dtype::F32).expect("Array::from_bytes failed for a");
    let b = Array::from_bytes(bytes, &shape, Dtype::F32).expect("Array::from_bytes failed for b");

    // Run on CPU — avoids Metal flakiness in parallel test runs.
    let c = add(&a, &b, Device::Cpu).expect("add failed");
    c.eval().expect("mlx_array_eval failed");

    let out_bytes = c.to_bytes().expect("to_bytes failed");
    let out = bytes_to_f32(&out_bytes);

    assert_eq!(
        out,
        vec![2.0f32, 4.0, 6.0, 8.0],
        "expected [2, 4, 6, 8], got {out:?}"
    );
}

/// GPU variant — ignored by default; add `-- --ignored` to run.
#[test]
#[ignore = "requires Metal GPU; run with `-- --ignored` in a GPU-capable environment"]
fn add_two_f32_arrays_gpu() {
    let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let bytes = f32_as_bytes(&input);
    let shape = [2i32, 2];
    let a = Array::from_bytes(bytes, &shape, Dtype::F32).unwrap();
    let b = Array::from_bytes(bytes, &shape, Dtype::F32).unwrap();
    let c = add(&a, &b, Device::Gpu).unwrap();
    c.eval().unwrap();
    let out = bytes_to_f32(&c.to_bytes().unwrap());
    assert_eq!(out, vec![2.0f32, 4.0, 6.0, 8.0]);
}

/// Smoke test for the blocking-thread GPU stream guard.
///
/// The generate entry points call `ensure_gpu_default_stream()` once at the
/// top of each tokio blocking-pool worker so MLX's per-thread CommandEncoder
/// map has a registered encoder before any `Array::eval()`. This test
/// runs that exact sequence on a freshly-spawned OS thread (the analog of a
/// blocking-pool worker that never called `mlx::core::new_stream`): establish
/// the stream, then materialise a GPU array.
///
/// Note: this asserts the guard is safe + idempotent off a worker thread and
/// that GPU eval succeeds there. It is NOT a strict negative regression — the
/// "no Stream(gpu, 0)" eval failure is MLX-version- and timing-dependent
/// (recent mlx-c may lazily register the encoder for the global default stream
/// on first access), so a bare fresh-thread eval does not reliably fault. The
/// real cross-path proof is a live serve exercising the speculative / image
/// generate paths.
#[test]
#[ignore = "requires Metal GPU; run with `-- --ignored` in a GPU-capable environment"]
fn gpu_default_stream_guard_idempotent_on_worker_thread() {
    let handle = std::thread::spawn(|| {
        // Establish the GPU default stream for THIS thread, exactly as the
        // blocking-thread generate entry points do before any materialisation.
        ensure_gpu_default_stream();
        // Idempotent: a second call from the same thread is a no-op.
        ensure_gpu_default_stream();

        let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes = f32_as_bytes(&input);
        let shape = [2i32, 2];
        let a = Array::from_bytes(bytes, &shape, Dtype::F32).unwrap();
        let b = Array::from_bytes(bytes, &shape, Dtype::F32).unwrap();
        let c = add(&a, &b, Device::Gpu).unwrap();
        c.eval().unwrap();
        bytes_to_f32(&c.to_bytes().unwrap())
    });
    let out = handle.join().expect("worker thread panicked");
    assert_eq!(out, vec![2.0f32, 4.0, 6.0, 8.0]);
}

#[test]
fn from_bytes_wrong_size_is_err() {
    let result = Array::from_bytes(&[0u8; 3], &[2, 2], Dtype::F32);
    assert!(result.is_err(), "expected Err for wrong byte count");
}

#[test]
fn array_debug_format() {
    let input: [f32; 4] = [0.0; 4];
    let a = Array::from_bytes(f32_as_bytes(&input), &[2, 2], Dtype::F32).unwrap();
    let dbg = format!("{a:?}");
    assert!(dbg.contains("F32"), "debug should mention dtype: {dbg}");
    assert!(dbg.contains('2'), "debug should mention shape dim: {dbg}");
}

#[test]
fn try_clone_works() {
    let input: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let a = Array::from_bytes(f32_as_bytes(&input), &[4], Dtype::F32).unwrap();
    let b = a.try_clone().expect("try_clone failed");
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
}

// ── multiply ─────────────────────────────────────────────────────────────

#[test]
fn multiply_f32() {
    let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b_data: [f32; 4] = [2.0, 3.0, 4.0, 5.0];
    let a = Array::from_bytes(f32_as_bytes(&a_data), &[4], Dtype::F32).unwrap();
    let b = Array::from_bytes(f32_as_bytes(&b_data), &[4], Dtype::F32).unwrap();
    let c = multiply(&a, &b, Device::Cpu).unwrap();
    c.eval().unwrap();
    let out = bytes_to_f32(&c.to_bytes().unwrap());
    assert_eq!(out, vec![2.0, 6.0, 12.0, 20.0]);
}

// ── divide ───────────────────────────────────────────────────────────────

#[test]
fn divide_f32() {
    let a_data: [f32; 4] = [4.0, 9.0, 16.0, 25.0];
    let b_data: [f32; 4] = [2.0, 3.0, 4.0, 5.0];
    let a = Array::from_bytes(f32_as_bytes(&a_data), &[4], Dtype::F32).unwrap();
    let b = Array::from_bytes(f32_as_bytes(&b_data), &[4], Dtype::F32).unwrap();
    let c = divide(&a, &b, Device::Cpu).unwrap();
    c.eval().unwrap();
    let out = bytes_to_f32(&c.to_bytes().unwrap());
    assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
}

// ── astype ───────────────────────────────────────────────────────────────

#[test]
fn astype_f32_to_f32() {
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[4], Dtype::F32).unwrap();
    let b = a.astype(Dtype::F32, Device::Cpu).unwrap();
    b.eval().unwrap();
    let out = bytes_to_f32(&b.to_bytes().unwrap());
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}

// ── reshape ──────────────────────────────────────────────────────────────

#[test]
fn reshape_basic() {
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[6], Dtype::F32).unwrap();
    let b = a.reshape(&[2, 3], Device::Cpu).unwrap();
    b.eval().unwrap();
    assert_eq!(b.shape(), vec![2, 3]);
    let out = bytes_to_f32(&b.to_bytes().unwrap());
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

// ── transpose ────────────────────────────────────────────────────────────

#[test]
fn transpose_2d() {
    // Input: [[1, 2, 3], [4, 5, 6]] shape [2, 3]
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2, 3], Dtype::F32).unwrap();
    let b = a.transpose(&[1, 0], Device::Cpu).unwrap();
    b.eval().unwrap();
    // Shape must be [3, 2] after transposing [2, 3].
    assert_eq!(b.shape(), vec![3, 2]);
    // Verify semantic correctness via matmul: [[1,2,3],[4,5,6]] @ [[1,2,3],[4,5,6]].T
    // = [[1,2,3],[4,5,6]] @ [[1,4],[2,5],[3,6]] = [[14, 32], [32, 77]]
    let c = matmul(&a, &b, Device::Cpu).unwrap();
    c.eval().unwrap();
    let out = bytes_to_f32(&c.to_bytes().unwrap());
    assert!(
        (out[0] - 14.0).abs() < 1e-4,
        "a@aT[0,0] expected 14, got {}",
        out[0]
    );
    assert!(
        (out[1] - 32.0).abs() < 1e-4,
        "a@aT[0,1] expected 32, got {}",
        out[1]
    );
    assert!(
        (out[2] - 32.0).abs() < 1e-4,
        "a@aT[1,0] expected 32, got {}",
        out[2]
    );
    assert!(
        (out[3] - 77.0).abs() < 1e-4,
        "a@aT[1,1] expected 77, got {}",
        out[3]
    );
}

// ── slice ─────────────────────────────────────────────────────────────────

#[test]
fn slice_1d() {
    let data: [f32; 6] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[6], Dtype::F32).unwrap();
    // Slice [1:4:1] → [20, 30, 40]
    let b = a.slice(&[1], &[4], &[1], Device::Cpu).unwrap();
    b.eval().unwrap();
    let out = bytes_to_f32(&b.to_bytes().unwrap());
    assert_eq!(out, vec![20.0, 30.0, 40.0]);
}

// ── slice_update ──────────────────────────────────────────────────────────

#[test]
fn slice_update_1d() {
    // src = [0, 0, 0, 0, 0], write [9, 8] at positions [1:3].
    // Expected: [0, 9, 8, 0, 0].
    let src_data: [f32; 5] = [0.0; 5];
    let upd_data: [f32; 2] = [9.0, 8.0];
    let src = Array::from_bytes(f32_as_bytes(&src_data), &[5], Dtype::F32).unwrap();
    let upd = Array::from_bytes(f32_as_bytes(&upd_data), &[2], Dtype::F32).unwrap();
    let res = src
        .slice_update(&upd, &[1], &[3], &[1], Device::Cpu)
        .unwrap();
    res.eval().unwrap();
    let out = bytes_to_f32(&res.to_bytes().unwrap());
    assert_eq!(out, vec![0.0, 9.0, 8.0, 0.0, 0.0]);
}

#[test]
fn slice_update_2d_kv_pattern() {
    // Simulate pre-allocated KV buffer: [1, 1, 4, 2] zeros, write [1, 1, 2, 2] at seq offset 1.
    let src_data: [f32; 8] = [0.0; 8];
    let upd_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let src = Array::from_bytes(f32_as_bytes(&src_data), &[1, 1, 4, 2], Dtype::F32).unwrap();
    let upd = Array::from_bytes(f32_as_bytes(&upd_data), &[1, 1, 2, 2], Dtype::F32).unwrap();
    let res = src
        .slice_update(
            &upd,
            &[0, 0, 1, 0],
            &[1, 1, 3, 2],
            &[1, 1, 1, 1],
            Device::Cpu,
        )
        .unwrap();
    res.eval().unwrap();
    let out = bytes_to_f32(&res.to_bytes().unwrap());
    // Row-major: [0, 0, 1, 2, 3, 4, 0, 0]
    assert_eq!(out, vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
}

// ── take ─────────────────────────────────────────────────────────────────

#[test]
fn take_axis0() {
    // Matrix 3×2, take rows [0, 2].
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[3, 2], Dtype::F32).unwrap();
    let idx_data: [i32; 2] = [0, 2];
    let idx_bytes =
        unsafe { std::slice::from_raw_parts(idx_data.as_ptr().cast::<u8>(), idx_data.len() * 4) };
    let idx = Array::from_bytes(idx_bytes, &[2], Dtype::I32).unwrap();
    let b = a.take(&idx, 0, Device::Cpu).unwrap();
    b.eval().unwrap();
    assert_eq!(b.shape(), vec![2, 2]);
    let out = bytes_to_f32(&b.to_bytes().unwrap());
    assert_eq!(out, vec![1.0, 2.0, 5.0, 6.0]);
}

// ── matmul ───────────────────────────────────────────────────────────────

#[test]
fn matmul_2x2() {
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let b_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
    let a = Array::from_bytes(f32_as_bytes(&a_data), &[2, 2], Dtype::F32).unwrap();
    let b = Array::from_bytes(f32_as_bytes(&b_data), &[2, 2], Dtype::F32).unwrap();
    let c = matmul(&a, &b, Device::Cpu).unwrap();
    c.eval().unwrap();
    let out = bytes_to_f32(&c.to_bytes().unwrap());
    assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
}

// ── softmax ───────────────────────────────────────────────────────────────

#[test]
fn softmax_sums_to_one() {
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[1, 4], Dtype::F32).unwrap();
    let b = softmax(&a, -1, Device::Cpu).unwrap();
    b.eval().unwrap();
    let out = bytes_to_f32(&b.to_bytes().unwrap());
    let sum: f32 = out.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum}");
}

// ── rms_norm ─────────────────────────────────────────────────────────────

#[test]
fn rms_norm_unit_weight() {
    // For weight=1.0 vector, rms_norm(x) = x / sqrt(mean(x^2) + eps).
    let x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let w_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let w = Array::from_bytes(f32_as_bytes(&w_data), &[4], Dtype::F32).unwrap();
    let out = rms_norm(&x, Some(&w), 1e-6, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    // rms(x) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // Expected: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
    let rms = (7.5_f32).sqrt();
    let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "rms_norm[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn rms_norm_no_weight() {
    // RMSNormNoScale: weight=None → plain rms normalization, no scale.
    let x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 4], Dtype::F32).unwrap();
    let out = rms_norm(&x, None, 1e-6, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    let rms = (7.5_f32).sqrt();
    let expected = [1.0 / rms, 2.0 / rms, 3.0 / rms, 4.0 / rms];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "rms_norm_no_weight[{i}]: got {got}, expected {exp}"
        );
    }
}

// ── gelu_tanh ─────────────────────────────────────────────────────────────

#[test]
fn gelu_tanh_near_zero() {
    // gelu(0) should be ~0.0
    let data: [f32; 4] = [0.0, 1.0, -1.0, 2.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[4], Dtype::F32).unwrap();
    let out = gelu_tanh(&a, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    // gelu(0) = 0, gelu(1) ≈ 0.841, gelu(-1) ≈ -0.159, gelu(2) ≈ 1.955
    assert!(
        vals[0].abs() < 1e-4,
        "gelu(0) should be ~0, got {}",
        vals[0]
    );
    assert!(
        vals[1] > 0.8 && vals[1] < 0.9,
        "gelu(1) should be ~0.84, got {}",
        vals[1]
    );
    assert!(
        vals[2] < 0.0 && vals[2] > -0.2,
        "gelu(-1) should be ~-0.16, got {}",
        vals[2]
    );
}

// ── silu ─────────────────────────────────────────────────────────────────

#[test]
fn silu_basic() {
    // silu(0) = 0, silu(1) = 1 / (1 + exp(-1)) ≈ 0.731
    let data: [f32; 2] = [0.0, 1.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2], Dtype::F32).unwrap();
    let out = silu(&a, Device::Cpu).unwrap();
    out.eval().unwrap();
    let vals = bytes_to_f32(&out.to_bytes().unwrap());
    assert!(vals[0].abs() < 1e-5, "silu(0) should be 0");
    assert!(
        (vals[1] - 0.731).abs() < 0.002,
        "silu(1) ≈ 0.731, got {}",
        vals[1]
    );
}

// ── argmax ────────────────────────────────────────────────────────────────

#[test]
fn argmax_last_axis() {
    let data: [f32; 6] = [1.0, 5.0, 3.0, 7.0, 2.0, 4.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2, 3], Dtype::F32).unwrap();
    let idx = argmax(&a, -1, Device::Cpu).unwrap();
    idx.eval().unwrap();
    let raw = idx.to_bytes().unwrap();
    // row 0: max at index 1 (value 5); row 1: max at index 0 (value 7)
    let vals: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals, vec![1, 0], "argmax: got {vals:?}");
}

// ── rope_with_freqs ───────────────────────────────────────────────────────

/// Smoke test for `rope_with_freqs`: check output shape, no NaN.
///
/// Numerical correctness is validated by the Gemma4 smoke probe end-to-end.
/// This test only asserts that the kernel runs, produces the right shape,
/// and does not NaN on a small synthetic input with well-formed freqs.
#[test]
fn rope_with_freqs_shape_no_nan() {
    // x: [batch=1, heads=1, seq=1, head_dim=4] (dims=4)
    // freqs: [dims/2=2] — one rotated pair + one inf pair
    let x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let x = Array::from_bytes(f32_as_bytes(&x_data), &[1, 1, 1, 4], Dtype::F32).unwrap();

    // freqs: first pair rotated at freq=1.0, second pair left untouched (inf).
    let freqs_data: [f32; 2] = [1.0, f32::INFINITY];
    let freqs = Array::from_bytes(f32_as_bytes(&freqs_data), &[2], Dtype::F32).unwrap();

    let out =
        rope_with_freqs(&x, 4, false, 1.0, 0, &freqs, Device::Cpu).expect("rope_with_freqs failed");
    out.eval().expect("eval failed");

    // Shape must be unchanged.
    assert_eq!(out.shape(), vec![1, 1, 1, 4], "shape changed");

    // No NaN.
    let bytes = out.to_bytes().expect("to_bytes failed");
    let vals = bytes_to_f32(&bytes);
    for (i, &v) in vals.iter().enumerate() {
        assert!(!v.is_nan(), "NaN at index {i}: {vals:?}");
    }
}

// ── expand_dims ───────────────────────────────────────────────────────────

#[test]
fn expand_dims_axis0() {
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[4], Dtype::F32).unwrap();
    let b = expand_dims(&a, 0, Device::Cpu).unwrap();
    b.eval().unwrap();
    assert_eq!(b.shape(), vec![1, 4]);
}

// ── scalar_f32 ───────────────────────────────────────────────────────────

#[test]
fn scalar_f32_roundtrip() {
    let s = scalar_f32(42.0);
    s.eval().unwrap();
    let bytes = s.to_bytes().unwrap();
    let val = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    assert!((val - 42.0).abs() < 1e-6, "scalar_f32 roundtrip: {val}");
}

// ── stream exhaustion regression ─────────────────────────────────────────
//
// Before the fix, each `with_stream` call issued `mlx_stream_new_device`
// which spawned a new OS thread. macOS caps per-process threads at ~2 048.
// 1 000 add calls × ~2 stream handles each → ~2 000 threads → EAGAIN.
// After the fix, `mlx_default_cpu_stream_new` reuses the existing thread;
// the 1 000-iteration loop stays well within OS limits.
#[test]
fn add_1000_iterations_no_thread_exhaustion() {
    let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let bytes = f32_as_bytes(&data);
    let shape = [4i32];

    let a = Array::from_bytes(bytes, &shape, Dtype::F32).expect("from_bytes a");
    let mut acc = Array::from_bytes(bytes, &shape, Dtype::F32).expect("from_bytes acc");

    for i in 0..1000 {
        acc = add(&acc, &a, Device::Cpu)
            .unwrap_or_else(|e| panic!("add failed at iteration {i}: {e}"));
        // Evaluate every 100 steps to materialise results and exercise eval path.
        if i % 100 == 99 {
            acc.eval()
                .unwrap_or_else(|e| panic!("eval failed at iteration {i}: {e}"));
        }
    }
    // Final eval + value check: started at [1,2,3,4], added [1,2,3,4] 1001 times.
    acc.eval().expect("final eval");
    let out = bytes_to_f32(&acc.to_bytes().expect("to_bytes"));
    // 1 initial + 1000 adds = 1001 × [1,2,3,4]
    let expected = [1001.0f32, 2002.0, 3003.0, 4004.0];
    for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1.0,
            "iteration 1000 result[{i}]: got {got}, expected {exp}"
        );
    }
}

fn mk(v: &[f32]) -> Array {
    Array::from_bytes(f32_as_bytes(v), &[v.len() as i32], Dtype::F32).unwrap()
}

/// The CPU analog of `gpu_default_stream_guard_idempotent_on_worker_thread`.
///
/// A tokio blocking-pool worker starts with no MLX stream context. Since MLX
/// 0.31/0.32 the default CPU/GPU streams and the CPU command encoders are
/// thread-local, so the generation entry points call `ensure_cpu_default_stream`
/// (and its GPU sibling) once per worker before building/evaluating any graph.
/// This runs that exact sequence on a freshly-spawned OS thread — the analog of
/// a blocking-pool worker — and asserts the guard is safe + idempotent and that
/// a CPU op *built and evaluated on that worker* succeeds.
///
/// Note (documented truth): this guard registers the *worker's own* default CPU
/// stream. It makes worker-built graphs eval cleanly, but it does NOT let a
/// worker evaluate an array whose ops were built on a *different* thread — that
/// is an upstream MLX limitation (streams from `new_stream` are usable only on
/// their thread of creation; see `cross_thread_eval_faults_documents_mlx_limit`).
#[test]
fn cpu_default_stream_guard_idempotent_on_worker_thread() {
    let out = std::thread::spawn(|| {
        // Establish the worker's default CPU stream, exactly as the generate
        // entry points do before any materialisation.
        ensure_cpu_default_stream();
        // Idempotent: a second call from the same thread is a no-op.
        ensure_cpu_default_stream();

        // Build AND evaluate on this worker thread — the safe, supported path.
        let c = add(
            &mk(&[1.0, 2.0, 3.0, 4.0]),
            &mk(&[1.0, 2.0, 3.0, 4.0]),
            Device::Cpu,
        )
        .unwrap();
        c.eval().unwrap();
        bytes_to_f32(&c.to_bytes().unwrap())
    })
    .join()
    .expect("worker thread panicked");
    assert_eq!(out, vec![2.0f32, 4.0, 6.0, 8.0]);
}

/// CI-runnable, GPU-free negative control isolating the **CPU guard alone**
/// (never calls `ensure_gpu_default_stream`) on a **scale-reduction** shaped
/// CPU pipeline — `square → max-reduce → divide` — the same op shape as the
/// K8V8 `exit_prefill` scale computation (`abs_max` reduction, then
/// `scale = abs_max / 127`) that this fix targets, so a pass here cannot be
/// explained away by the GPU guard mattering instead of the CPU one.
///
/// Honest scope note: MLX's own `default_stream(Device)` lazily self-registers
/// a fresh per-thread stream + `CommandEncoder` on first use
/// (`mlx/stream.cpp::default_stream` → `new_stream` → `cpu::new_stream`), so a
/// worker thread that **builds and evaluates its own graph** already succeeds
/// without any guard call — confirmed empirically (this exact op shape run
/// with no guard on a spawned worker also passes). A true
/// "faults-without-guard, succeeds-with-guard" negative control is therefore
/// not constructible for this same-thread shape: the only fault class this fix
/// addresses is genuinely cross-thread (an array built on one thread, eval'd on
/// another — see `cross_thread_eval_faults_documents_mlx_limit`, which the
/// guard does **not** cure either, since it registers the eval thread's *own*
/// stream, not the foreign one). What this test *does* prove, CI-runnably and
/// without any GPU dependency: the CPU guard alone (no GPU guard in the call
/// path) is sufficient for a worker thread to build and evaluate a
/// reduction-shaped CPU graph — the exact shape `exit_prefill` needs.
#[test]
fn cpu_guard_alone_handles_scale_reduction_on_worker_thread() {
    let out = std::thread::spawn(|| {
        // CPU guard only — deliberately no `ensure_gpu_default_stream()` call
        // anywhere in this thread, so a pass cannot be attributed to the GPU
        // guard.
        ensure_cpu_default_stream();

        // square → max-reduce → divide: the same op shape as the K8V8
        // exit_prefill scale computation (abs_max reduction, then
        // scale = abs_max / divisor), built AND evaluated on this worker.
        let x = mk(&[1.0, -3.0, 2.0, -4.0]);
        let squared = multiply(&x, &x, Device::Cpu).unwrap();
        let reduced = max_axis(&squared, 0, Device::Cpu).unwrap(); // scalar: 16.0
        let divisor = mk(&[2.0]);
        let scale = divide(&reduced, &divisor, Device::Cpu).unwrap();
        scale.eval().unwrap();
        bytes_to_f32(&scale.to_bytes().unwrap())
    })
    .join()
    .expect("worker thread panicked");
    assert_eq!(out, vec![8.0f32]);
}

/// Executable documentation of the *true* fault class behind the
/// "There is no Stream(cpu, N) in current thread" crash.
///
/// An MLX array whose ops were built on thread A carries that thread's stream
/// affinity; evaluating it from thread B throws, because MLX ≥0.31 default
/// streams (from `new_stream`) are usable only on their thread of creation and
/// the CPU command-encoder map is thread-local. `ensure_cpu_default_stream` on
/// the eval thread does NOT fix this (it registers a *different* stream). The
/// only cures are (a) evaluate the array on its building thread (materialised
/// arrays are then safe to read anywhere), or (b) an MLX `thread_unsafe` stream,
/// which mlx-c 0.6.0 does not expose.
///
/// Ignored by default: it deliberately provokes an upstream error and is
/// timing/version sensitive; it exists to pin the mechanism, not to gate CI.
#[test]
#[ignore = "documents upstream MLX cross-thread stream limitation; provokes an error on purpose"]
fn cross_thread_eval_faults_documents_mlx_limit() {
    // Establish this (main/test) thread's default CPU stream, then build a lazy
    // op here so it is bound to this thread's stream.
    let warm = add(&mk(&[1.0]), &mk(&[1.0]), Device::Cpu).unwrap();
    warm.eval().unwrap();
    let c = add(&mk(&[1.0, 2.0]), &mk(&[3.0, 4.0]), Device::Cpu).unwrap();

    // Evaluate the here-built array on a different thread → upstream fault.
    let res = std::thread::spawn(move || c.eval().map_err(|e| format!("{e}")))
        .join()
        .expect("worker thread panicked");
    assert!(
        res.is_err(),
        "expected a cross-thread stream fault; got Ok — MLX behaviour changed"
    );
    let msg = res.unwrap_err();
    assert!(
        msg.contains("no Stream"),
        "expected a 'There is no Stream(...)' fault, got: {msg}"
    );
}
