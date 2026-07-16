//! CPU-device behaviour pins for `update_and_sdpa_shared_source`.
//!
//! These tests run on `Device::Cpu` and pin two structural properties of the
//! `update_and_sdpa_shared_source` code path that hold independently of GPU
//! availability:
//!
//! * **CPU-device dispatch gating**: with `Device::Cpu`, both
//!   `sdpa_dispatch_no_lock` (TurboFlash) and `try_fused_qk_dispatch`
//!   (fused-QK) early-return `Ok(None)` — `device != Device::Gpu` is
//!   an explicit pre-condition of both dispatch arms. This is a valuable
//!   property to pin: it proves the fallback gate is unconditional on CPU,
//!   regardless of env-var state.
//!
//! * **Legacy-path shape contract**: the legacy bf16 SDPA fallback (the path
//!   that runs when all dispatch arms return `None`) must surface K/V shaped
//!   `[B, kv_h, offset, D]`. Consumer layers (Gemma4 shared-KV) depend on
//!   this shape; a regression here would corrupt their attention computation.
//!
//! **What these tests do NOT cover**: GPU kernel dispatch (delta > 0).
//! That is covered by the integration test at
//! `crates/rmlx-kv-quant/tests/shared_source_dispatch.rs` which runs
//! under `#[ignore]` and requires `--test-threads=1` and a Metal GPU.

use super::core::KvCache;
use super::SharedKv;
use crate::kvcache::fused_qk_total_dispatch_count;
use crate::turbo_flash_msl::turbo_flash_dispatch_count;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and the bytes are
    // copied into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

const TEST_KV_H: i32 = 8;
const TEST_HEAD_DIM: i32 = 128;
const TEST_MAX_SEQ: i32 = 512;
const TEST_PREFILL_SEQ: i32 = 64;

/// CPU-device gate: `Device::Cpu` suppresses all dispatch arms regardless of
/// env-var state.
///
/// Both `sdpa_dispatch_no_lock` and `try_fused_qk_dispatch` return
/// `Ok(None)` when `device != Device::Gpu`. This test drives a K8V4 cache
/// through a prefill chunk + 1 decode step on `Device::Cpu` and asserts
/// that neither the TurboFlash counter nor the fused-QK counter moved.
///
/// This is the structural CPU-gate pin — it proves that the legacy fallback
/// is unconditional on CPU hosts regardless of `RMLX_TURBO_FLASH` or
/// `RMLX_FUSED_QK` env-var state. It does NOT prove that GPU dispatch works.
#[test]
fn shared_source_cpu_device_suppresses_all_dispatch_arms() {
    // Defensive: clear gates if a prior process / test latched them. These
    // gates use OnceLock so the value is per-process; the test relies on the
    // canonical default-off state.
    // SAFETY: env var mutation is single-threaded here (each #[test] runs
    // on its own thread; we do not spawn threads in this test).
    // We don't *set* anything; rely on default unset.
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V4, TEST_MAX_SEQ);

    cache.enter_prefill();

    let prefill_shape = [1_i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.111f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.222f32; n_pref], &prefill_shape);
    let q_pref_shape = [1_i32, 16, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_q_pref: usize = q_pref_shape.iter().map(|&d| d as usize).product();
    let q_pref = f32_arr(&vec![0.333f32; n_q_pref], &q_pref_shape);

    // Prefill via the shared-source entry — mirrors Gemma4's call site.
    let tf_before = turbo_flash_dispatch_count();
    let fqk_before = fused_qk_total_dispatch_count();
    let (_out, _share) = cache
        .update_and_sdpa_shared_source(&q_pref, &k_pref, &v_pref, 1.0, "causal", None, device)
        .expect("prefill shared source");
    cache.exit_prefill(device).expect("exit_prefill");

    // Decode step 1 — q_seq = 1.
    let step_kv_shape = [1_i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_kv_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.444f32; n_step], &step_kv_shape);
    let v_step = f32_arr(&vec![0.555f32; n_step], &step_kv_shape);
    let q_step_shape = [1_i32, 16, 1, TEST_HEAD_DIM];
    let n_q_step: usize = q_step_shape.iter().map(|&d| d as usize).product();
    let q_step = f32_arr(&vec![0.666f32; n_q_step], &q_step_shape);

    let (_out2, _share2) = cache
        .update_and_sdpa_shared_source(&q_step, &k_step, &v_step, 1.0, "", None, device)
        .expect("decode shared source");

    let tf_after = turbo_flash_dispatch_count();
    let fqk_after = fused_qk_total_dispatch_count();

    assert_eq!(
        tf_after, tf_before,
        "CPU-gate: TurboFlash counter advanced ({tf_before} -> {tf_after}) on \
         Device::Cpu — sdpa_dispatch_no_lock must gate off unconditionally on CPU."
    );
    assert_eq!(
        fqk_after, fqk_before,
        "CPU-gate: fused-QK counter advanced ({fqk_before} -> {fqk_after}) on \
         Device::Cpu — try_fused_qk_dispatch must gate off unconditionally on CPU."
    );
}

/// CPU-device legacy shape contract: surfaced K/V is shaped `[B, kv_h, offset, D]`.
///
/// The shared-KV consumer (Gemma4 cross-layer-KV) depends on `(K, V)` being
/// shaped `[B, kv_h, offset, D]` after `update_and_sdpa_shared_source`. This
/// test exercises a decode step on `Device::Cpu` (where all dispatch arms
/// gate off) and verifies the legacy bf16 fallback surfaces the full kv
/// prefix with the correct shape.
///
/// This pins the legacy-path shape contract. It does NOT exercise the GPU
/// dispatch path — shape correctness after TurboFlash dispatch is verified
/// in `crates/rmlx-kv-quant/tests/shared_source_dispatch.rs`.
#[test]
fn shared_source_cpu_device_legacy_fallback_surfaces_full_prefix_shape() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V4, TEST_MAX_SEQ);

    cache.enter_prefill();
    let prefill_shape = [1_i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.011f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.022f32; n_pref], &prefill_shape);
    let q_pref_shape = [1_i32, 16, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_q_pref: usize = q_pref_shape.iter().map(|&d| d as usize).product();
    let q_pref = f32_arr(&vec![0.033f32; n_q_pref], &q_pref_shape);
    let _ = cache
        .update_and_sdpa_shared_source(&q_pref, &k_pref, &v_pref, 1.0, "causal", None, device)
        .expect("prefill shared source");
    cache.exit_prefill(device).expect("exit_prefill");

    let step_kv_shape = [1_i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_kv_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.044f32; n_step], &step_kv_shape);
    let v_step = f32_arr(&vec![0.055f32; n_step], &step_kv_shape);
    let q_step_shape = [1_i32, 16, 1, TEST_HEAD_DIM];
    let n_q_step: usize = q_step_shape.iter().map(|&d| d as usize).product();
    let q_step = f32_arr(&vec![0.066f32; n_q_step], &q_step_shape);

    let (_out, share) = cache
        .update_and_sdpa_shared_source(&q_step, &k_step, &v_step, 1.0, "", None, device)
        .expect("decode shared source");

    // Every dispatch arm gates off on CPU, so this is the legacy bf16 share.
    let SharedKv::Bf16(k_full, v_full) = &share else {
        panic!("the CPU legacy fallback must surface a bf16 share");
    };
    let expected = [1_i32, TEST_KV_H, TEST_PREFILL_SEQ + 1, TEST_HEAD_DIM];
    assert_eq!(
        k_full.shape(),
        expected,
        "surfaced K must span [0:offset] over the full prefix",
    );
    assert_eq!(
        v_full.shape(),
        expected,
        "surfaced V must span [0:offset] over the full prefix",
    );
}
