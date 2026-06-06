// Rotor fused-QK fast path: integration tests covering all 6 rotor variants,
// using the fused-QK shadow split with QJL fallback gate.
//
// `#[ignore]`-gated because they need the GPU; run via:
//   cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- \
//       --ignored --test-threads=1
//
// CLAUDE.md hard rule 8 (single MLX process): the test claims no port — the
// integration runner naturally serialises tests within one process. Run
// after `pkill ... && rm -f /tmp/rmlx.<port>.claim`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    unsafe_code,
    missing_docs
)]
//! Rotor fused-QK dispatch integration test.

use rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
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

/// Drive prefill + one decode step on `codec`; expect the rotor fused-QK
/// kernel to dispatch (delta > 0). Cosine vs bf16 SDPA is informational
/// — rotor's lossy codec means a 0.99 cosine floor is not appropriate
/// here; we assert the kernel fired and the result is finite.
fn run_rotor_dispatch_test(codec: KvQuant, name: &str) {
    let device = Device::Gpu;
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads: i32 = kv_h * heads_per_kv;
    let head_dim: i32 = 128;
    let prefill_seq: i32 = 64;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let mut cache = KvCache::with_quant_max_seq(codec, 4096);
    cache.enter_prefill();
    let prefill_k_shape = [b, kv_h, prefill_seq, head_dim];
    let n_k: usize = prefill_k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_k, 0xB1B2_C3D4_E5F6_0020);
    let v_data = lcg_data(n_k, 0xB1B2_C3D4_E5F6_0021);
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
    let q_data = lcg_data(q_n, 0xB1B2_C3D4_E5F6_0022);
    let q_arr = make_f32_array(&q_data, &decode_q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let new_k_shape = [b, kv_h, 1, head_dim];
    let new_n: usize = new_k_shape.iter().map(|&d| d as usize).product();
    let new_k_data = lcg_data(new_n, 0xB1B2_C3D4_E5F6_0023);
    let new_v_data = lcg_data(new_n, 0xB1B2_C3D4_E5F6_0024);
    let new_k = make_f32_array(&new_k_data, &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&new_v_data, &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");

    let before = fused_qk_total_dispatch_count();
    let out_res = cache.update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device);
    let after = fused_qk_total_dispatch_count();
    let out = out_res.unwrap_or_else(|e| panic!("{name}: update_and_sdpa failed: {e}"));
    let delta = after - before;
    eprintln!("{name}: rotor fused-QK dispatch delta = {delta}");
    assert!(
        delta >= 1,
        "{name}: rotor decode with RMLX_FUSED_QK=1 must dispatch the rotor fused-QK kernel at least once (delta={delta})"
    );
    out.eval().expect("eval rotor SDPA output");
    let _ = out.to_bytes().expect("rotor SDPA output materialised");
}

fn set_fused_qk_on() {
    // SAFETY: process-global env var, single-threaded test enforced.
    unsafe {
        std::env::set_var("RMLX_FUSED_QK", "1");
    }
    unsafe {
        std::env::set_var("RMLX_FUSED_QK_MIN", "8");
    }
    // Ensure rotor QJL is OFF — the kernel does not consume the QJL
    // residual. Note: the rotor_qjl default is ON when the env var is
    // absent; we must explicitly set `0` to disable so the fused-QK
    // dispatch passes the QJL fallback gate.
    unsafe {
        std::env::set_var("RMLX_ROTOR_QJL", "0");
    }
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor3_sym_fused_qk_dispatch() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    run_rotor_dispatch_test(KvQuant::Rotor3Sym, "Rotor3Sym");
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor4_sym_fused_qk_dispatch() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    run_rotor_dispatch_test(KvQuant::Rotor4Sym, "Rotor4Sym");
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_3_fused_qk_dispatch() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    run_rotor_dispatch_test(KvQuant::RotorKOnly3, "RotorKOnly3");
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_4_fused_qk_dispatch() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    run_rotor_dispatch_test(KvQuant::RotorKOnly4, "RotorKOnly4");
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_asym_3_fused_qk_dispatch() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    run_rotor_dispatch_test(
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        "RotorK3Asym(v=q4_g64)",
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_k_asym_4_fused_qk_dispatch() {
    if skip_if_no_gpu() {
        return;
    }
    set_fused_qk_on();
    run_rotor_dispatch_test(
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        "RotorK4Asym(v=q4_g64)",
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test rotor_fused_qk_dispatch -- --ignored --test-threads=1"]
fn rotor_qjl_on_falls_back_to_legacy_sdpa() {
    // When RMLX_ROTOR_QJL=1, `try_fused_qk_dispatch` must short-circuit
    // (return Ok(None)) and the legacy bf16 SDPA path takes over. The
    // dispatch counter must NOT increment.
    if skip_if_no_gpu() {
        return;
    }
    // SAFETY: process-global env var, single-threaded test enforced.
    unsafe {
        std::env::set_var("RMLX_FUSED_QK", "1");
    }
    unsafe {
        std::env::set_var("RMLX_FUSED_QK_MIN", "8");
    }
    unsafe {
        std::env::set_var("RMLX_ROTOR_QJL", "1");
    }
    let device = Device::Gpu;
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads: i32 = kv_h * heads_per_kv;
    let head_dim: i32 = 128;
    let prefill_seq: i32 = 64;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let mut cache = KvCache::with_quant_max_seq(KvQuant::Rotor3Sym, 4096);
    cache.enter_prefill();
    let prefill_k_shape = [b, kv_h, prefill_seq, head_dim];
    let n_k: usize = prefill_k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_k, 0xC0DE_BEEF_0001);
    let v_data = lcg_data(n_k, 0xC0DE_BEEF_0002);
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

    let decode_q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = decode_q_shape.iter().map(|&d| d as usize).product();
    let q_arr = make_f32_array(&lcg_data(q_n, 0xC0DE_BEEF_0003), &decode_q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");
    let new_k_shape = [b, kv_h, 1, head_dim];
    let new_n: usize = new_k_shape.iter().map(|&d| d as usize).product();
    let new_k = make_f32_array(&lcg_data(new_n, 0xC0DE_BEEF_0004), &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&lcg_data(new_n, 0xC0DE_BEEF_0005), &new_k_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");

    let before = fused_qk_total_dispatch_count();
    let out = cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("update_and_sdpa");
    let after = fused_qk_total_dispatch_count();
    let delta = after - before;
    assert_eq!(
        delta, 0,
        "QJL-on: rotor fused-QK must NOT dispatch when QJL is enabled (delta={delta}); fallback to legacy SDPA expected"
    );
    out.eval().expect("eval SDPA output");
    // Clean up env for subsequent tests.
    unsafe {
        std::env::remove_var("RMLX_ROTOR_QJL");
    }
}
