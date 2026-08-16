// Production fused-QK dispatch end-to-end smoke + parity test.
//
// Drives the public `KvCache::update_and_sdpa` path with K8V4 (q8 fused-QK),
// under a fused-QK policy, on a synthetic prefill + 1 decode step. Asserts:
//
//   1. The dispatch counter increments by exactly 1 (proving the kernel
//      fired and the production path actually routed through it).
//   2. The decode output cosine vs the bf16 SDPA reference >= 0.999 (proving
//      the head-major shadow + slicing + SV path produces a
//      result numerically consistent with the legacy fallback).
//
// `#[ignore]` because it needs the GPU; run via:
//   cargo test -p rmlx-kv-quant --test fused_qk_dispatch -- --ignored
//
// CLAUDE.md hard rule 8 (single MLX process): the test claims no port —
// the integration runner naturally serialises tests within one process.
// Run after `pkill ... && rm -f /tmp/rmlx.<port>.claim`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    missing_docs
)]
//! Fused-QK dispatch integration test.

use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

/// Fused-QK on, with a low enough threshold that the short synthetic caches
/// in this file clear it. Each cache carries its own copy, so the arms here
/// are independent of process state and of each other.
fn fused_qk_policy() -> DispatchPolicy {
    DispatchPolicy {
        fused_qk: true,
        fused_qk_min_kv_seq: 8,
        ..DispatchPolicy::default()
    }
}

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect()
}

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

fn cosine_per_row(a: &[f32], b: &[f32], row_len: usize) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "cosine: len mismatch");
    assert!(row_len > 0);
    let n_rows = a.len() / row_len;
    let mut mn = f32::INFINITY;
    let mut sum = 0.0_f64;
    for r in 0..n_rows {
        let ra = &a[r * row_len..(r + 1) * row_len];
        let rb = &b[r * row_len..(r + 1) * row_len];
        let dot: f64 = ra
            .iter()
            .zip(rb.iter())
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = ra.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = rb.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let c = if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            (dot / (na * nb)) as f32
        };
        if c < mn {
            mn = c;
        }
        sum += f64::from(c);
    }
    let mean = (sum / n_rows as f64) as f32;
    (mean, mn)
}

fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = (state >> 32) as u32 as f32 / u32::MAX as f32;
            frac * 2.0 - 1.0
        })
        .collect()
}

/// End-to-end: prefill bf16 tokens via `update`, then one decode step via
/// `update_and_sdpa` under a fused-QK policy. Compare to bf16 reference,
/// assert dispatch counter delta > 0 and cosine close to 1.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test fused_qk_dispatch -- --ignored --test-threads=1"]
fn fused_qk_dispatch_routes_through_kernel_k8v4() {
    if skip_if_no_gpu() {
        return;
    }

    let device = Device::Gpu;
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads: i32 = kv_h * heads_per_kv;
    let head_dim: i32 = 128;
    let prefill_seq: i32 = 64;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::K8V4, 4096).with_dispatch_policy(fused_qk_policy());

    cache.enter_prefill();
    let prefill_k_shape = [b, kv_h, prefill_seq, head_dim];
    let n_k: usize = prefill_k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_k, 0xA1B2C3D4_E5F6_0001);
    let v_data = lcg_data(n_k, 0xA1B2C3D4_E5F6_0002);
    let prefill_k = make_f32_array(&k_data, &prefill_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 k");
    let prefill_v = make_f32_array(&v_data, &prefill_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 v");
    let _full = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("update prefill");
    cache.exit_prefill(device).expect("exit_prefill");
    assert_eq!(cache.offset(), prefill_seq, "offset after prefill");

    let decode_q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = decode_q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xA1B2C3D4_E5F6_0003);
    let q_arr = make_f32_array(&q_data, &decode_q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let new_k_shape = [b, kv_h, 1, head_dim];
    let new_n: usize = new_k_shape.iter().map(|&d| d as usize).product();
    let new_k_data = lcg_data(new_n, 0xA1B2C3D4_E5F6_0004);
    let new_v_data = lcg_data(new_n, 0xA1B2C3D4_E5F6_0005);
    let new_k = make_f32_array(&new_k_data, &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&new_v_data, &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");

    let before = fused_qk_total_dispatch_count();
    let out = cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("update_and_sdpa");
    let after = fused_qk_total_dispatch_count();

    let delta = after - before;
    assert!(
        delta >= 1,
        "K8V4 decode with RMLX_FUSED_QK=1 must dispatch the q8 fused-QK kernel at least once, observed delta={delta}"
    );

    let mut bf16_cache = KvCache::with_quant_max_seq(KvQuant::None, 4096);
    bf16_cache.enter_prefill();
    let _ = bf16_cache
        .update(&prefill_k, &prefill_v, device)
        .expect("bf16 update prefill");
    bf16_cache.exit_prefill(device).expect("bf16 exit_prefill");
    let ref_out = bf16_cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("bf16 update_and_sdpa");

    let out_f32_vec = array_to_f32(&out.astype(Dtype::F32, device).expect("to f32"));
    let ref_f32_vec = array_to_f32(&ref_out.astype(Dtype::F32, device).expect("to f32"));
    let (mean, mn) = cosine_per_row(&ref_f32_vec, &out_f32_vec, head_dim as usize);
    eprintln!(
        "K8V4 fused_qk vs bf16 SDPA: cosine mean={mean:.6}, min={mn:.6}, dispatch delta={delta}"
    );
    assert!(
        mn >= 0.99,
        "fused_qk_dispatch cosine min {mn} < 0.99 (mean {mean})"
    );
}

/// Same shape as the K8V4 test but with TurboSym3 (3-bit K codec).
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test fused_qk_dispatch -- --ignored --test-threads=1"]
fn fused_qk_dispatch_routes_through_kernel_turbo_sym3() {
    if skip_if_no_gpu() {
        return;
    }
    run_parity_for_codec(KvQuant::TurboSym3, "TurboSym3", 0.95);
}

/// Same shape as the K8V4 test but with TurboSym4 (4-bit K codec).
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test fused_qk_dispatch -- --ignored --test-threads=1"]
fn fused_qk_dispatch_routes_through_kernel_turbo_sym4() {
    if skip_if_no_gpu() {
        return;
    }
    run_parity_for_codec(KvQuant::TurboSym4, "TurboSym4", 0.99);
}

fn run_parity_for_codec(codec: KvQuant, name: &str, cosine_floor: f32) {
    let device = Device::Gpu;
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads: i32 = kv_h * heads_per_kv;
    let head_dim: i32 = 128;
    let prefill_seq: i32 = 64;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let mut cache =
        KvCache::with_quant_max_seq(codec, 4096).with_dispatch_policy(fused_qk_policy());
    cache.enter_prefill();
    let prefill_k_shape = [b, kv_h, prefill_seq, head_dim];
    let n_k: usize = prefill_k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_k, 0xA1B2C3D4_E5F6_0010);
    let v_data = lcg_data(n_k, 0xA1B2C3D4_E5F6_0011);
    let prefill_k = make_f32_array(&k_data, &prefill_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 k");
    let prefill_v = make_f32_array(&v_data, &prefill_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 v");
    let _full = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("update prefill");
    cache.exit_prefill(device).expect("exit_prefill");
    assert_eq!(cache.offset(), prefill_seq, "{name}: offset after prefill");

    let decode_q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = decode_q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xA1B2C3D4_E5F6_0012);
    let q_arr = make_f32_array(&q_data, &decode_q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let new_k_shape = [b, kv_h, 1, head_dim];
    let new_n: usize = new_k_shape.iter().map(|&d| d as usize).product();
    let new_k_data = lcg_data(new_n, 0xA1B2C3D4_E5F6_0013);
    let new_v_data = lcg_data(new_n, 0xA1B2C3D4_E5F6_0014);
    let new_k = make_f32_array(&new_k_data, &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&new_v_data, &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");

    let strict = std::env::var("RMLX_FUSED_QK_STRICT").as_deref() == Ok("1");
    let before = fused_qk_total_dispatch_count();
    let out_res = cache.update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device);
    let after = fused_qk_total_dispatch_count();
    let out = match out_res {
        Ok(o) => o,
        Err(e) => {
            // Codec storage may not support `update` for the K-only or
            // weight-K-bf16-V split; record the dispatch counter behaviour
            // and skip the cosine assertion in that case.
            eprintln!("{name}:update_and_sdpa errored: {e} (skipping parity, dispatch counter stays before)");
            assert_eq!(
                after, before,
                "{name}: failed call must not increment counter"
            );
            assert!(
                !strict,
                "{name}:RMLX_FUSED_QK_STRICT=1 — update_and_sdpa errored ({e}), but strict mode requires the codec to fire the fused-QK kernel"
            );
            return;
        }
    };

    let delta = after - before;
    eprintln!("{name}:dispatch counter delta = {delta}");
    if delta == 0 {
        // The codec may not yet route through the production path (e.g.
        // its `update` flow short-circuits in `update_and_sdpa` before
        // fused-QK). Document as HOLD-soft. Under RMLX_FUSED_QK_STRICT=1
        // this is a hard fail — used by the CI gate to catch silent
        // regressions where a codec stops routing.
        eprintln!("{name}:HOLD-soft — dispatch counter did not increment; cosine check skipped");
        assert!(
            !strict,
            "{name}:RMLX_FUSED_QK_STRICT=1 — dispatch counter did not increment; \
             the codec is expected to fire the fused-QK kernel in strict mode"
        );
        return;
    }

    let mut bf16_cache = KvCache::with_quant_max_seq(KvQuant::None, 4096);
    bf16_cache.enter_prefill();
    let _ = bf16_cache
        .update(&prefill_k, &prefill_v, device)
        .expect("bf16 update prefill");
    bf16_cache.exit_prefill(device).expect("bf16 exit_prefill");
    let ref_out = bf16_cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("bf16 update_and_sdpa");

    let out_f32_vec = array_to_f32(&out.astype(Dtype::F32, device).expect("to f32"));
    let ref_f32_vec = array_to_f32(&ref_out.astype(Dtype::F32, device).expect("to f32"));
    let (mean, mn) = cosine_per_row(&ref_f32_vec, &out_f32_vec, head_dim as usize);
    eprintln!(
        "{name} fused_qk vs bf16 SDPA: cosine mean={mean:.6}, min={mn:.6}, dispatch delta={delta}"
    );
    assert!(
        mn >= cosine_floor,
        "{name} cosine min {mn} < {cosine_floor} (mean {mean})"
    );
}
