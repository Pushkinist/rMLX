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

// ── to_bytes over non-contiguous views ────────────────────────────────────
//
// Every array below shares its parent's allocation and differs from it only in
// strides, so a reader that walks the data pointer linearly returns the
// parent's leading elements under the view's shape — right length, right
// dtype, wrong values, no error. `slice_1d` above cannot cover this: a rank-1
// stride-1 slice is dense by construction because its offset lands in the data
// pointer.

#[test]
fn to_bytes_of_a_strided_slice_reads_the_slice() {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let a = Array::from_bytes(f32_as_bytes(&data), &[1, 2, 4, 2], Dtype::F32).unwrap();
    // Take the first two of four sequence positions in both heads.
    let view = a
        .slice(&[0, 0, 0, 0], &[1, 2, 2, 2], &[1, 1, 1, 1], Device::Cpu)
        .unwrap();
    let out = bytes_to_f32(&view.to_bytes().unwrap());
    assert_eq!(
        out,
        vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
        "head 1 must start at element 8; [4..8] is the parent's tail of head 0"
    );
}

#[test]
fn to_bytes_of_a_rank2_slice_reads_the_window() {
    let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
    let a = Array::from_bytes(f32_as_bytes(&data), &[3, 3], Dtype::F32).unwrap();
    // Two leading columns of every row.
    let view = a.slice(&[0, 0], &[3, 2], &[1, 1], Device::Cpu).unwrap();
    let out = bytes_to_f32(&view.to_bytes().unwrap());
    assert_eq!(out, vec![0.0, 1.0, 3.0, 4.0, 6.0, 7.0]);
}

#[test]
fn to_bytes_of_a_transpose_reads_the_permuted_order() {
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2, 3], Dtype::F32).unwrap();
    let view = a.transpose(&[1, 0], Device::Cpu).unwrap();
    assert_eq!(view.shape(), vec![3, 2]);
    let out = bytes_to_f32(&view.to_bytes().unwrap());
    assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn to_bytes_of_an_attention_v_transpose_reads_head_major_order() {
    // The shape every attention block hands to the KV cache: `[b, seq, kv_h,
    // head_dim]` permuted to `[b, kv_h, seq, head_dim]`. Dense at seq == 1 —
    // the permuted axis has extent 1 — and strided for every longer step.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let a = Array::from_bytes(f32_as_bytes(&data), &[1, 3, 2, 2], Dtype::F32).unwrap();
    let view = a.transpose(&[0, 2, 1, 3], Device::Cpu).unwrap();
    assert_eq!(view.shape(), vec![1, 2, 3, 2]);
    let out = bytes_to_f32(&view.to_bytes().unwrap());
    assert_eq!(
        out,
        vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0]
    );
}

#[test]
fn to_bytes_of_a_broadcast_reads_the_expanded_shape() {
    // A broadcast view is the one case where the linear read is not merely
    // wrong: its logical size exceeds the elements the parent actually owns.
    let data: [f32; 3] = [1.0, 2.0, 3.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[1, 3], Dtype::F32).unwrap();
    let view = broadcast_to(&a, &[2, 3], Device::Cpu).unwrap();
    let out = bytes_to_f32(&view.to_bytes().unwrap());
    assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
}

#[test]
fn reshape_of_a_transposed_view_is_relaid_out_and_says_so() {
    // `to_bytes` skips the relayout on the strength of the layout flag, so the
    // flag has to be right about the one op that quietly materialises a copy:
    // MLX reshapes a non-row-contiguous input by copying it dense.
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2, 3], Dtype::F32).unwrap();
    let flat = a
        .transpose(&[1, 0], Device::Cpu)
        .unwrap()
        .reshape(&[6], Device::Cpu)
        .unwrap();
    flat.eval().unwrap();
    assert!(flat.is_row_contiguous().unwrap());
    assert_eq!(
        bytes_to_f32(&flat.to_bytes().unwrap()),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

/// GPU variant — the relayout runs on the CPU stream while the view it reads
/// was evaluated on the Metal one, which is the one thing the CPU cases above
/// cannot show.
#[test]
#[ignore = "requires Metal GPU; run with `-- --ignored` in a GPU-capable environment"]
fn to_bytes_of_a_gpu_transpose_reads_the_permuted_order() {
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2, 3], Dtype::F32).unwrap();
    let view = a.transpose(&[1, 0], Device::Gpu).unwrap();
    view.eval().unwrap();
    let out = bytes_to_f32(&view.to_bytes().unwrap());
    assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn to_bytes_of_an_empty_array_is_empty() {
    // The load-path warmup evaluates a `[0]` array through this method.
    let a = Array::from_bytes(&[], &[0], Dtype::F32).unwrap();
    assert!(a.to_bytes().unwrap().is_empty());
}

#[test]
fn layout_flag_classifies_views_once_they_are_evaluated() {
    // The flag describes a materialised buffer, so it is only meaningful after
    // `eval` — which is the order `to_bytes` reads it in.
    let data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Array::from_bytes(f32_as_bytes(&data), &[2, 3], Dtype::F32).unwrap();
    a.eval().unwrap();
    assert!(a.is_row_contiguous().unwrap());

    let view = a.transpose(&[1, 0], Device::Cpu).unwrap();
    view.eval().unwrap();
    assert!(!view.is_row_contiguous().unwrap());

    let dense = view.contiguous(Device::Cpu).unwrap();
    dense.eval().unwrap();
    assert!(
        dense.is_row_contiguous().unwrap(),
        "the relayout to_bytes falls back to must itself be readable linearly"
    );
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

/// Executable statement of the mechanism the evaluation lock exists for: on the
/// linked MLX the CPU command-encoder map is **process-global**, so an array
/// built — and therefore stream-bound — on one thread evaluates perfectly well
/// on another.
///
/// That is the whole problem. A map every thread can reach is shared mutable
/// state, and MLX fills it with no synchronisation
/// (`mlx/backend/cpu/encoder.cpp::get_command_encoder`), which is why
/// evaluation has to be serialised on our side — see
/// `concurrent_first_eval_across_threads_does_not_corrupt_encoder_map`.
///
/// This also pins *which* upstream model is live. MLX 0.32.0 makes that map
/// `thread_local` and throws "There is no Stream(cpu, N) in current thread."
/// for exactly this shape, so if the MLX pin moves, this fails loudly here
/// instead of quietly invalidating the reasoning in `ensure_cpu_default_stream`.
#[test]
fn cross_thread_eval_resolves_through_the_process_global_encoder_map() {
    // Bind this thread's default CPU stream and put its encoder in the map.
    let warm = add(&mk(&[1.0]), &mk(&[1.0]), Device::Cpu).unwrap();
    warm.eval().unwrap();

    // Build here, so the graph carries this thread's stream, then evaluate it
    // on a thread that never established a CPU stream of its own.
    let c = add(&mk(&[1.0, 2.0]), &mk(&[3.0, 4.0]), Device::Cpu).unwrap();
    let result = std::thread::spawn(move || {
        c.eval().map_err(|e| format!("{e}"))?;
        c.to_bytes().map_err(|e| format!("{e}"))
    })
    .join()
    .expect("worker thread panicked");

    let bytes = result.unwrap_or_else(|e| {
        panic!(
            "cross-thread CPU eval failed: {e}\n\
             The encoder map is no longer process-global — MLX 0.32.0 makes it \
             thread_local and throws here. Recheck ensure_cpu_default_stream's \
             rationale and the evaluation lock against the new pin."
        )
    });
    assert_eq!(bytes_to_f32(&bytes), vec![4.0f32, 6.0]);
}

/// The deterministic half of the evaluation-lock gate: `with_eval_lock` really
/// does exclude.
///
/// `make check-eval-lock` proves every evaluation FFI call is *written* inside
/// a `with_eval_lock` closure — a lexical property. It cannot tell whether the
/// lock actually locks; a `with_eval_lock` that took no lock would satisfy it
/// completely. This closes that half, and unlike the burst reproducer it is
/// deterministic: two threads, no MLX, no 400-thread cost, and it fails every
/// time if mutual exclusion is lost rather than one run in twelve.
///
/// The oracle is independent of the lock's implementation — a flag raised and
/// cleared *inside* the critical section, with each entrant asserting it found
/// the section empty. It shares no arithmetic with the code under test.
#[test]
fn with_eval_lock_serialises_concurrent_callers() {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static OCCUPIED: AtomicBool = AtomicBool::new(false);
    static OVERLAPS: AtomicUsize = AtomicUsize::new(0);

    // Long enough that a lost lock overlaps essentially every iteration, short
    // enough that the whole test stays well under a second.
    const ITERS: usize = 200;
    const HOLD: std::time::Duration = std::time::Duration::from_micros(50);

    let worker = || {
        for _ in 0..ITERS {
            with_eval_lock(|| {
                // Entering: the section must have been empty.
                if OCCUPIED.swap(true, Ordering::SeqCst) {
                    OVERLAPS.fetch_add(1, Ordering::SeqCst);
                }
                std::thread::sleep(HOLD);
                // Leaving: and must still have been ours.
                if !OCCUPIED.swap(false, Ordering::SeqCst) {
                    OVERLAPS.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    };

    let a = std::thread::spawn(worker);
    let b = std::thread::spawn(worker);
    a.join().expect("thread a panicked");
    b.join().expect("thread b panicked");

    assert_eq!(
        OVERLAPS.load(Ordering::SeqCst),
        0,
        "two threads were inside with_eval_lock at the same time — the evaluation \
         lock is not excluding, so every MLX eval FFI call under it is unprotected"
    );
}

/// **A reproducer, not a gate — and deliberately out of the default run.**
/// Without the lock it fails about **1 run in 12** (15 in 180, measured), so
/// eleven times in twelve it would go green with the defect fully present.
/// Worse, that figure was measured the way `make eval-lock-stress` runs it:
/// alone, against a genuinely cold encoder map. Inside a normal `cargo test`
/// the map is already warm from every other MLX-touching test, so its real
/// detection chance there is lower still and has never been measured — while
/// its cost is exact and certain (412 threads peak against 4 without it, ~436
/// still resident when the suite ends, in a binary whose test order is
/// nondeterministic). A gate whose cost is measured and whose benefit is not
/// does not belong in `make ci`.
///
/// The gates for this defect are `make check-eval-lock` (every eval FFI call
/// is made under the lock) and
/// `with_eval_lock_serialises_concurrent_callers` (the lock actually excludes),
/// both deterministic. This test is the end-to-end demonstration that the two
/// of them together prevent the real crash; run it with
/// `make eval-lock-stress`, which drives it across fresh processes because the
/// map only starts cold once per process.
///
/// The linked MLX resolves a CPU stream's `CommandEncoder` through a
/// process-global `std::unordered_map<int, CommandEncoder>` that it fills
/// **lazily, on first evaluation, with no synchronisation**
/// (`mlx/backend/cpu/encoder.cpp::get_command_encoder`). Its default-CPU-stream
/// storage is per-thread (`mlx/stream.cpp::default_stream_storage`), so every
/// thread that evaluates a CPU graph mints its *own* stream index and therefore
/// performs its *own* insert into that one shared map. Two inserts in flight
/// together rehash the map underneath a third thread's bucket walk and the
/// process takes SIGSEGV — the whole test binary dies and libtest reports no
/// failing test, because no test failed.
///
/// `cargo test` runs each test on its own OS thread, so any crate with many
/// MLX-touching tests reproduces exactly this shape. This test compresses it
/// into one burst against a cold map: the map rehashes at every prime bucket
/// count it grows through, and starting from empty is what puts *all* of those
/// rehashes inside a single window with hundreds of inserts in flight.
///
/// It passes only because `Array::eval` / `Array::async_eval` evaluate under
/// `with_eval_lock`. Drop that and it fails as SIGSEGV, SIGTRAP, or an
/// infinite spin on a bucket chain that became circular — all three were
/// observed.
///
/// Scope: the **CPU** evaluation path only. Concurrent *GPU* evaluation is not
/// exercised (those tests carry `#[ignore]` and run serialised), nor are races
/// inside lazy graph *construction*, which never reaches `eval`.
#[test]
// Not a GPU test — CPU only. Ignored because it is probabilistic and costs
// ~412 threads; `make eval-lock-stress` is its runner and passes `--ignored`.
#[ignore = "probabilistic reproducer, ~412 threads — run via `make eval-lock-stress`"]
#[allow(
    clippy::needless_collect,
    reason = "the collect is the concurrency: it spawns every thread before the first join, \
              whereas a lazy iterator would spawn and join one at a time and never overlap them"
)]
fn concurrent_first_eval_reproducer() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Barrier};

    // Cost of this number, measured rather than estimated: the binary peaks at
    // 412 threads running this test versus 4 without it, and MLX 0.31.2 has no
    // stream-reclaim path, so the streams minted here — and the OS threads
    // behind them — persist for the life of the test binary. The whole crate
    // suite peaks at 446 and is still holding ~436 when it finishes, i.e. every
    // later test in this binary runs in a 400-thread process.
    //
    // Headroom against `sysctl kern.num_taskthreads` (16384 on this machine) is
    // ~40x. That is the real ceiling; the ~2 048 figure this crate used to cite
    // is unverified, and against it the margin would be only ~5x. Both are
    // survivable, neither is "two orders of magnitude" — an earlier revision of
    // this comment claimed that and was wrong in the unsafe direction.
    const THREADS: usize = 400;

    // A barrier alone leaves the threads spread over the condvar wake-up. Park
    // them on a spinning flag instead: they are already running when it flips,
    // so the inserts land together.
    let gate = Arc::new(AtomicBool::new(false));
    let ready = Arc::new(Barrier::new(THREADS + 1));

    let workers: Vec<_> = (0..THREADS)
        .map(|i| {
            let gate = Arc::clone(&gate);
            let ready = Arc::clone(&ready);
            std::thread::spawn(move || {
                let v = i as f32;
                // Build first. MLX ops are lazy, so this only mints this
                // thread's CPU stream — and stream creation takes an MLX mutex,
                // which would otherwise stagger the threads apart before they
                // ever reach the unsynchronised part.
                let sum = add(&mk(&[v, v]), &mk(&[v, v]), Device::Cpu).expect("cpu add failed");
                ready.wait();
                while !gate.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                sum.eval().expect("cpu eval failed");
                bytes_to_f32(&sum.to_bytes().expect("to_bytes failed"))
            })
        })
        .collect();

    ready.wait();
    gate.store(true, Ordering::Release);

    for (i, worker) in workers.into_iter().enumerate() {
        let got = worker.join().expect("worker thread panicked");
        let want = (i as f32) * 2.0;
        assert_eq!(
            got,
            vec![want, want],
            "thread {i} read back the wrong values — encoder map or stream state was corrupted"
        );
    }
}

// ── peak-memory bracket ──────────────────────────────────────────────────

/// The derived quantities are pure arithmetic over the three raw counters, so
/// they are pinned here against hand-built readings — no allocator, no device.
/// A GPU-side exercise of the real bracket lives in
/// `crates/rmlx-kv-quant/src/q8_msl_tests.rs`.
#[test]
fn peak_reading_headroom_excludes_bytes_already_live() {
    let r = PeakReading {
        peak_bytes: 900,
        live_at_open_bytes: 500,
        live_at_close_bytes: 700,
        reset_ok: true,
    };
    // 500 bytes were already resident when the bracket opened; the region is
    // charged only for the 400 it added on top.
    assert_eq!(r.headroom_bytes(), 400);
    // 200 of those 400 were released again before close.
    assert_eq!(r.transient_bytes(), 200);
    assert!(r.observed_allocation());
}

#[test]
fn peak_reading_saturates_when_region_allocated_nothing() {
    // Reset leaves the mark at 0 and nothing raised it, while 500 bytes stayed
    // live throughout. Both deltas must floor at 0 rather than wrap.
    let r = PeakReading {
        peak_bytes: 0,
        live_at_open_bytes: 500,
        live_at_close_bytes: 500,
        reset_ok: true,
    };
    assert_eq!(r.headroom_bytes(), 0);
    assert_eq!(r.transient_bytes(), 0);
    assert!(
        !r.observed_allocation(),
        "a region that allocated nothing must not satisfy the anti-vacuous check"
    );
}

#[test]
fn peak_reading_transient_is_zero_when_nothing_was_freed() {
    // Everything the region peaked at is still live at close.
    let r = PeakReading {
        peak_bytes: 900,
        live_at_open_bytes: 500,
        live_at_close_bytes: 900,
        reset_ok: true,
    };
    assert_eq!(r.headroom_bytes(), 400);
    assert_eq!(r.transient_bytes(), 0);
}

/// The anti-vacuous predicate must key off the region, not the process.
///
/// MLX updates the mark as `peak = max(peak, active)` on every allocation, and
/// `active` is the whole live count. So after a reset, one allocation anywhere
/// in the process lifts `peak_bytes` to at least the resident total — which in
/// `rmlx baseline` is gigabytes of weights. A predicate of `peak_bytes > 0`
/// would be true in every real process and would certify nothing.
#[test]
fn observed_allocation_is_about_the_region_not_the_process() {
    // The peak never rose above what was already live: this region did not
    // allocate, even though `peak_bytes` is a large non-zero number.
    let untouched = PeakReading {
        peak_bytes: 4_000_000_000,
        live_at_open_bytes: 4_000_000_000,
        live_at_close_bytes: 4_000_000_000,
        reset_ok: true,
    };
    assert_eq!(untouched.headroom_bytes(), 0);
    assert!(
        !untouched.observed_allocation(),
        "peak == live_at_open means the region added nothing; a `peak_bytes > 0` \
         predicate would call this an observed allocation"
    );

    // One byte above the open live count is an observed allocation.
    let touched = PeakReading {
        peak_bytes: 4_000_000_001,
        ..untouched
    };
    assert_eq!(touched.headroom_bytes(), 1);
    assert!(touched.observed_allocation());
}

/// A reset that did not happen leaves the peak process-lifetime, so the deltas
/// describe the process rather than the region. They must report nothing rather
/// than a large, stable, plausible-looking number that a harness would diff.
#[test]
fn failed_reset_reports_nothing_rather_than_a_plausible_number() {
    let r = PeakReading {
        peak_bytes: 9_000_000_000,
        live_at_open_bytes: 4_000_000_000,
        live_at_close_bytes: 4_000_000_000,
        reset_ok: false,
    };
    assert!(!r.measurable());
    assert_eq!(
        r.headroom_bytes(),
        0,
        "an unscoped peak must not surface as 5 GB of region headroom"
    );
    assert_eq!(r.transient_bytes(), 0);
    assert!(!r.observed_allocation());

    // The identical counters with a successful reset are a real measurement.
    let ok = PeakReading {
        reset_ok: true,
        ..r
    };
    assert!(ok.measurable());
    assert_eq!(ok.headroom_bytes(), 5_000_000_000);
}
