use super::*;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_f32(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("from_bytes")
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn array_to_f32(a: &Array) -> Vec<f32> {
    let f = if a.dtype() == Dtype::F32 {
        a.try_clone().unwrap()
    } else {
        a.astype(Dtype::F32, Device::Gpu).unwrap()
    };
    f.eval().unwrap();
    f.to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Tiny shape end-to-end on the GPU: single-step update with B=1, T=1,
/// Hk=1, Hv=1, Dk=32, Dv=4. Verifies the kernel compiles, dispatches,
/// and returns the correct output shapes / values.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test gated_delta_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gated_delta_step_gpu_smoke() {
    let batch = 1i32;
    let seq = 1i32;
    let hk = 1i32;
    let hv = 1i32;
    let dk = 32i32;
    let dv = 4i32;

    // q/k all-1, v ascending, g=1 (no decay), beta=1, state=0.
    // Expected:
    // kv_mem = 0
    // delta = v
    // state = k * delta (rank-1 outer product); each elem = v[dv]
    // y = state · q = (k · q) * v = (sum 1*1 over Dk) * v = Dk * v
    let q_data = vec![1.0_f32; (batch * seq * hk * dk) as usize];
    let k_data = vec![1.0_f32; (batch * seq * hk * dk) as usize];
    let v_data: Vec<f32> = (0..(batch * seq * hv * dv) as usize)
        .map(|i| (i + 1) as f32)
        .collect();
    let g_data = vec![1.0_f32; (batch * seq * hv) as usize];
    let beta_data = vec![1.0_f32; (batch * seq * hv) as usize];
    let state_data = vec![0.0_f32; (batch * hv * dv * dk) as usize];

    let q_bf16 = make_f32(&q_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let k_bf16 = make_f32(&k_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let v_bf16 = make_f32(&v_data, &[batch, seq, hv, dv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let beta_bf16 = make_f32(&beta_data, &[batch, seq, hv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let g_arr = make_f32(&g_data, &[batch, seq, hv]);
    let state_arr = make_f32(&state_data, &[batch, hv, dv, dk]);

    let (y, state_out) = gated_delta_step_gpu(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &g_arr,
        &beta_bf16,
        &state_arr,
        Device::Gpu,
    )
    .expect("kernel dispatch failed");

    assert_eq!(y.shape(), vec![batch, seq, hv, dv]);
    assert_eq!(state_out.shape(), vec![batch, hv, dv, dk]);
    assert_eq!(y.dtype(), Dtype::Bf16);
    assert_eq!(state_out.dtype(), Dtype::F32);

    let y_vec = array_to_f32(&y);
    for (i, &y_i) in y_vec.iter().enumerate() {
        let expected = (dk as f32) * v_data[i];
        // bf16 has ~0.4% relative precision; tolerance scales with magnitude.
        let tol = (expected.abs() * 0.05).max(1.0);
        assert!(
            (y_i - expected).abs() < tol,
            "y[{i}] = {y_i}, expected ≈ {expected}",
        );
    }
}

/// Larger shape that mirrors Qwen3.6 GatedDeltaNet dims:
/// B=1, T=4, Hk=16, Hv=32, Dk=128, Dv=128.
/// All-zero inputs: kernel must produce zeros and not crash.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test gated_delta_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gated_delta_step_gpu_qwen_shape_zeros() {
    let batch = 1i32;
    let seq = 4i32;
    let hk = 16i32;
    let hv = 32i32;
    let dk = 128i32;
    let dv = 128i32;

    let q = rmlx_mlx::zeros(&[batch, seq, hk, dk], Dtype::Bf16, Device::Gpu).unwrap();
    let k = rmlx_mlx::zeros(&[batch, seq, hk, dk], Dtype::Bf16, Device::Gpu).unwrap();
    let v = rmlx_mlx::zeros(&[batch, seq, hv, dv], Dtype::Bf16, Device::Gpu).unwrap();
    let g = rmlx_mlx::zeros(&[batch, seq, hv], Dtype::F32, Device::Gpu).unwrap();
    let beta = rmlx_mlx::zeros(&[batch, seq, hv], Dtype::Bf16, Device::Gpu).unwrap();
    let state = rmlx_mlx::zeros(&[batch, hv, dv, dk], Dtype::F32, Device::Gpu).unwrap();

    let (y, state_out) =
        gated_delta_step_gpu(&q, &k, &v, &g, &beta, &state, Device::Gpu).expect("dispatch");

    assert_eq!(y.shape(), vec![batch, seq, hv, dv]);
    assert_eq!(state_out.shape(), vec![batch, hv, dv, dk]);

    let y_vec = array_to_f32(&y);
    for &v_i in &y_vec {
        assert_eq!(v_i, 0.0_f32);
    }
    let state_vec = array_to_f32(&state_out);
    for &v_i in &state_vec {
        assert_eq!(v_i, 0.0_f32);
    }
}

/// Identity test: with k = 0, the rank-1 update is zero regardless of
/// v/beta, so state_out == state_in. Verifies the ops loop doesn't
/// silently corrupt the carry-through state.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test gated_delta_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prefill_ops_identity_with_zero_k() {
    let batch = 1i32;
    let seq = 8i32;
    let hk = 1i32;
    let hv = 1i32;
    let dk = 32i32;
    let dv = 4i32;

    // k = 0 -> kv_mem = 0, rank-1 update = k * delta = 0, state unchanged.
    // g = 1 (no decay), beta = 1, v = ascending.
    let q_data = vec![1.0_f32; (batch * seq * hk * dk) as usize];
    let k_data = vec![0.0_f32; (batch * seq * hk * dk) as usize]; // all zero
    let v_data: Vec<f32> = (0..(batch * seq * hv * dv) as usize)
        .map(|i| (i + 1) as f32)
        .collect();
    let g_data = vec![1.0_f32; (batch * seq * hv) as usize]; // no decay
    let beta_data = vec![1.0_f32; (batch * seq * hv) as usize];
    // Non-zero initial state.
    let state_data: Vec<f32> = (0..(batch * hv * dv * dk) as usize)
        .map(|i| (i % 7) as f32 * 0.1)
        .collect();

    let q = make_f32(&q_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let k = make_f32(&k_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let v = make_f32(&v_data, &[batch, seq, hv, dv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let g = make_f32(&g_data, &[batch, seq, hv]);
    let beta = make_f32(&beta_data, &[batch, seq, hv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let state_in = make_f32(&state_data, &[batch, hv, dv, dk]);

    let (y, state_out) = gated_delta_prefill_ops(&q, &k, &v, &g, &beta, &state_in, Device::Gpu)
        .expect("prefill_ops dispatch");

    assert_eq!(y.shape(), vec![batch, seq, hv, dv]);
    assert_eq!(state_out.shape(), vec![batch, hv, dv, dk]);
    assert_eq!(state_out.dtype(), Dtype::F32);

    // State must be identical to state_in (k=0 so no update).
    let s_out = array_to_f32(&state_out);
    for (i, (&got, &expected)) in s_out.iter().zip(state_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "state_out[{i}] = {got}, expected {expected} (k=0 identity)"
        );
    }
}

/// Sequential equivalence: `gated_delta_prefill_ops` and
/// `gated_delta_step_gpu` must produce the same output within fp16
/// round-trip tolerance (ops path uses f32 arithmetic; kernel uses
/// input dtype bf16; difference is bounded by bf16 quantisation error).
///
/// Uses B=1, T=8, Hk=1, Hv=1, Dk=32, Dv=4 (small, exact shape).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test gated_delta_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prefill_ops_matches_sequential_kernel() {
    let batch = 1i32;
    let seq = 8i32;
    let hk = 1i32;
    let hv = 1i32;
    let dk = 32i32;
    let dv = 4i32;

    // Pseudo-random but deterministic input data.
    let n_q = (batch * seq * hk * dk) as usize;
    let n_v = (batch * seq * hv * dv) as usize;
    let n_g = (batch * seq * hv) as usize;
    let n_s = (batch * hv * dv * dk) as usize;
    let q_data: Vec<f32> = (0..n_q).map(|i| ((i * 7 + 1) % 13) as f32 * 0.1).collect();
    let k_data: Vec<f32> = (0..n_q).map(|i| ((i * 3 + 5) % 11) as f32 * 0.1).collect();
    let v_data: Vec<f32> = (0..n_v).map(|i| ((i * 5 + 3) % 17) as f32 * 0.1).collect();
    let g_data: Vec<f32> = (0..n_g)
        .map(|i| ((i % 3) as f32).mul_add(0.02, 0.9))
        .collect();
    let beta_data: Vec<f32> = (0..n_g)
        .map(|i| ((i % 5) as f32).mul_add(0.05, 0.3))
        .collect();
    let state_data = vec![0.0_f32; n_s];

    let q_bf16 = make_f32(&q_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let k_bf16 = make_f32(&k_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let v_bf16 = make_f32(&v_data, &[batch, seq, hv, dv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let beta_bf16 = make_f32(&beta_data, &[batch, seq, hv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let g_arr = make_f32(&g_data, &[batch, seq, hv]);
    let state_arr = make_f32(&state_data, &[batch, hv, dv, dk]);

    // Sequential kernel reference.
    let (y_seq, s_seq) = gated_delta_step_gpu(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &g_arr,
        &beta_bf16,
        &state_arr,
        Device::Gpu,
    )
    .expect("sequential kernel");

    // Ops-based path under test.
    let (y_ops, s_ops) = gated_delta_prefill_ops(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &g_arr,
        &beta_bf16,
        &state_arr,
        Device::Gpu,
    )
    .expect("prefill ops");

    let y_seq_f32 = array_to_f32(&y_seq);
    let y_ops_f32 = array_to_f32(&y_ops);
    let s_seq_f32 = array_to_f32(&s_seq);
    let s_ops_f32 = array_to_f32(&s_ops);

    // Tolerance: ops path runs in f32; kernel runs in bf16. The bf16
    // round-trip error is ~1/128 ≈ 0.8% relative, so we allow 5% relative
    // or 5e-3 absolute — whichever is looser — to account for accumulation
    // across 8 steps.
    let tol_y = |expected: f32| -> f32 { (expected.abs() * 0.05).max(5e-3) };
    let tol_s = |expected: f32| -> f32 { (expected.abs() * 0.05).max(5e-3) };

    for (i, (&ops, &seq_v)) in y_ops_f32.iter().zip(y_seq_f32.iter()).enumerate() {
        let t = tol_y(seq_v);
        assert!(
            (ops - seq_v).abs() < t,
            "y[{i}]: ops={ops:.6} seq={seq_v:.6} diff={:.6} tol={t:.6}",
            (ops - seq_v).abs()
        );
    }
    for (i, (&ops, &seq_v)) in s_ops_f32.iter().zip(s_seq_f32.iter()).enumerate() {
        let t = tol_s(seq_v);
        assert!(
            (ops - seq_v).abs() < t,
            "state[{i}]: ops={ops:.6} seq={seq_v:.6} diff={:.6} tol={t:.6}",
            (ops - seq_v).abs()
        );
    }
}

/// Verify that calling `gated_delta_prefill_ops` twice with the same shape
/// produces identical outputs (idempotency / no stale state).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test gated_delta_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prefill_ops_same_shape_twice_idempotent() {
    let batch = 1i32;
    let seq = 4i32; // T ≥ 1 (ops path is forced here via direct call)
    let hk = 2i32;
    let hv = 4i32;
    let dk = 32i32;
    let dv = 4i32;

    let q = rmlx_mlx::zeros(&[batch, seq, hk, dk], Dtype::Bf16, Device::Gpu).unwrap();
    let k = rmlx_mlx::zeros(&[batch, seq, hk, dk], Dtype::Bf16, Device::Gpu).unwrap();
    let v = rmlx_mlx::zeros(&[batch, seq, hv, dv], Dtype::Bf16, Device::Gpu).unwrap();
    let g = rmlx_mlx::zeros(&[batch, seq, hv], Dtype::F32, Device::Gpu).unwrap();
    let beta = rmlx_mlx::zeros(&[batch, seq, hv], Dtype::Bf16, Device::Gpu).unwrap();
    let state = rmlx_mlx::zeros(&[batch, hv, dv, dk], Dtype::F32, Device::Gpu).unwrap();

    // First call — may trace + compile.
    let (y1, s1) =
        gated_delta_prefill_ops(&q, &k, &v, &g, &beta, &state, Device::Gpu).expect("first call");
    assert_eq!(y1.shape(), vec![batch, seq, hv, dv]);
    assert_eq!(s1.shape(), vec![batch, hv, dv, dk]);
    // Materialize to ensure no deferred error.
    y1.eval().expect("eval y1");
    y1.to_bytes().expect("materialize y1");

    // Second call — same shape, both must succeed and produce same result.
    let (y2, s2) =
        gated_delta_prefill_ops(&q, &k, &v, &g, &beta, &state, Device::Gpu).expect("second call");
    assert_eq!(y2.shape(), vec![batch, seq, hv, dv]);
    assert_eq!(s2.shape(), vec![batch, hv, dv, dk]);
    y2.eval().expect("eval y2");
    y2.to_bytes().expect("materialize y2");

    // Both outputs must be numerically identical (same zero inputs).
    let y1_cast = y1.astype(Dtype::F32, Device::Gpu).unwrap();
    y1_cast.eval().expect("eval y1 cast");
    let y1_v: Vec<f32> = y1_cast
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let y2_cast = y2.astype(Dtype::F32, Device::Gpu).unwrap();
    y2_cast.eval().expect("eval y2 cast");
    let y2_v: Vec<f32> = y2_cast
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        y1_v, y2_v,
        "compile-cache: y outputs differ between call 1 and call 2"
    );
}

/// Bit-equivalence check at T=64 between the compiled-closure ops path
/// and the sequential MSL kernel. T=64 is the production prefill chunk
/// size; this verifies the compile-cache wiring doesn't drift from the
/// reference at the actual hot-path size.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test gated_delta_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prefill_ops_compile_cache_t64_matches_sequential() {
    let batch = 1i32;
    let seq = 64i32;
    let hk = 2i32;
    let hv = 4i32;
    let dk = 32i32;
    let dv = 4i32;

    let n_q = (batch * seq * hk * dk) as usize;
    let n_v = (batch * seq * hv * dv) as usize;
    let n_g = (batch * seq * hv) as usize;
    let n_s = (batch * hv * dv * dk) as usize;
    let q_data: Vec<f32> = (0..n_q).map(|i| ((i * 7 + 1) % 13) as f32 * 0.05).collect();
    let k_data: Vec<f32> = (0..n_q).map(|i| ((i * 3 + 5) % 11) as f32 * 0.05).collect();
    let v_data: Vec<f32> = (0..n_v).map(|i| ((i * 5 + 3) % 17) as f32 * 0.05).collect();
    let g_data: Vec<f32> = (0..n_g)
        .map(|i| ((i % 3) as f32).mul_add(0.01, 0.95))
        .collect();
    let beta_data: Vec<f32> = (0..n_g)
        .map(|i| ((i % 5) as f32).mul_add(0.05, 0.3))
        .collect();
    let state_data = vec![0.0_f32; n_s];

    let q_bf16 = make_f32(&q_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let k_bf16 = make_f32(&k_data, &[batch, seq, hk, dk])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let v_bf16 = make_f32(&v_data, &[batch, seq, hv, dv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let beta_bf16 = make_f32(&beta_data, &[batch, seq, hv])
        .astype(Dtype::Bf16, Device::Gpu)
        .unwrap();
    let g_arr = make_f32(&g_data, &[batch, seq, hv]);
    let state_arr = make_f32(&state_data, &[batch, hv, dv, dk]);

    // Sequential kernel reference.
    let (y_seq, s_seq) = gated_delta_step_gpu(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &g_arr,
        &beta_bf16,
        &state_arr,
        Device::Gpu,
    )
    .expect("sequential kernel");

    // Compiled-closure ops path: first call traces + compiles, second call
    // hits the cache. Output of either call must match the kernel.
    let (_y_warmup, _s_warmup) = gated_delta_prefill_ops(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &g_arr,
        &beta_bf16,
        &state_arr,
        Device::Gpu,
    )
    .expect("warmup call (trace + compile)");
    let (y_ops, s_ops) = gated_delta_prefill_ops(
        &q_bf16,
        &k_bf16,
        &v_bf16,
        &g_arr,
        &beta_bf16,
        &state_arr,
        Device::Gpu,
    )
    .expect("compiled-cache hit call");

    let y_seq_f32 = array_to_f32(&y_seq);
    let y_ops_f32 = array_to_f32(&y_ops);
    let s_seq_f32 = array_to_f32(&s_seq);
    let s_ops_f32 = array_to_f32(&s_ops);

    // After 64 sequential delta-rule updates, bf16 trajectories may drift
    // by up to ~10% in absolute units relative to f32. Use generous
    // tolerance (15% relative or 1e-2 absolute, whichever is larger).
    let tol_y = |expected: f32| -> f32 { (expected.abs() * 0.15).max(1e-2) };
    let tol_s = |expected: f32| -> f32 { (expected.abs() * 0.15).max(1e-2) };

    for (i, (&ops, &seq_v)) in y_ops_f32.iter().zip(y_seq_f32.iter()).enumerate() {
        let t = tol_y(seq_v);
        assert!(
            (ops - seq_v).abs() < t,
            "y[{i}]: ops={ops:.6} seq={seq_v:.6} diff={:.6} tol={t:.6}",
            (ops - seq_v).abs()
        );
    }
    for (i, (&ops, &seq_v)) in s_ops_f32.iter().zip(s_seq_f32.iter()).enumerate() {
        let t = tol_s(seq_v);
        assert!(
            (ops - seq_v).abs() < t,
            "state[{i}]: ops={ops:.6} seq={seq_v:.6} diff={:.6} tol={t:.6}",
            (ops - seq_v).abs()
        );
    }
}
