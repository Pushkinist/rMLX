// Decode-path dtype contract: a KV cache must hand back attention output in
// the dtype its queries came in.
//
// WHY THIS EXISTS
// ---------------
// An MSL dispatcher declares its kernel output dtype explicitly
// (`add_output_shape(..., Dtype::F32)`), because online-softmax accumulation
// wants f32 internally. If it then returns that array without restoring the
// caller's dtype, MLX does exactly what it is designed to do: the f32
// attention output promotes the residual add, and the promotion propagates
// through the next layer's norm, its weight GEMV and every elementwise op —
// the whole decode graph re-instantiates in f32. Nothing errors, nothing
// warns; the only symptoms are throughput and non-bit-exactness.
//
// The sweep is driven off `ALL_KV_QUANTS`, so a codec added to the enum is
// covered here the moment it is added to that list — the coverage cannot
// silently stop at the codecs that existed when this file was written.
//
// NON-VACUITY
// -----------
// Every codec falls back to a bf16 path when its fused kernel does not
// dispatch, and a bf16 path trivially returns bf16. A pass therefore means
// nothing unless the fused kernels actually ran, so the `all kernels on` arm
// asserts a non-zero dispatch delta for the TurboFlash counter (the path this
// contract was broken on) and prints the per-arm totals for the rest.
//
// `#[ignore]`: needs the Metal GPU. Run via:
//   cargo test -p rmlx-kv-quant --test kv_decode_dtype_contract -- \
//       --ignored --test-threads=1 --nocapture
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::indexing_slicing,
    missing_docs
)]
//! Decode-path dtype contract sweep.

use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::iso_flash_decode_msl::{
    iso3_flash_decode_dispatch_count, iso4_flash_decode_dispatch_count,
};
use rmlx_kv_quant::iso_flash_decode_symv_msl::{
    iso3_symv_flash_decode_dispatch_count, iso4_symv_flash_decode_dispatch_count,
};
use rmlx_kv_quant::q8_fused_qk_msl::q8_fused_qk_dispatch_count;
use rmlx_kv_quant::rotor_flash_decode_msl::{
    rotor3_flash_decode_dispatch_count, rotor4_flash_decode_dispatch_count,
};
use rmlx_kv_quant::rotor_flash_decode_symv_msl::{
    rotor3_symv_flash_decode_dispatch_count, rotor4_symv_flash_decode_dispatch_count,
};
use rmlx_kv_quant::rotor_fused_qk_msl::{
    rotor3_fused_qk_dispatch_count, rotor4_fused_qk_dispatch_count,
};
use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
use rmlx_kv_quant::turbo_k3_fused_qk_msl::turbo_k3_fused_qk_dispatch_count;
use rmlx_kv_quant::turbo_k4_fused_qk_msl::turbo_k4_fused_qk_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant, ALL_KV_QUANTS};
use rmlx_mlx::{Array, Device, Dtype};

const B: i32 = 1;
const KV_H: i32 = 2;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;
/// 128 is the one head_dim every codec in the enum accepts: a power of two
/// (Hadamard / tree reductions), a multiple of 32 (tq4 / planar groups) and in
/// the {128, 256} set the TurboFlash kernel is wired for.
const HEAD_DIM: i32 = 128;
const PREFILL: i32 = 64;
const MAX_SEQ: i32 = 1024;

/// The dtype every arch drives its attention with. The contract is that this
/// is also the dtype that comes back.
const ACT_DTYPE: Dtype = Dtype::Bf16;

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

/// LCG pseudo-random data — same constants as the other integration tests so
/// reproducer seeds round-trip across reports.
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

fn bf16_tensor(shape: &[i32], seed: u64, device: Device) -> Array {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    make_f32_array(&lcg_data(n, seed), shape)
        .astype(ACT_DTYPE, device)
        .expect("astype activation dtype")
}

/// Every fused kernel a `KvCache` can dispatch, on, with the size thresholds
/// dropped so they fire on the short synthetic cache here. This is the arm that
/// exercises the MSL dispatchers; the default-policy arm covers the generic
/// paths those dispatchers replace.
///
/// `sparse_attn` is the one gate left off, and it is structural rather than an
/// omission: that path is not dispatched by `KvCache` at all. It is driven from
/// `rmlx-models` with a `HeadBudgets` table no cache-level sweep supplies, so
/// turning the flag on here would change nothing. Its dtype contract is pinned
/// where it is reachable — `rmlx_models::kv_cache::attention_dispatch`'s
/// `sparse_attn_dispatch_returns_the_query_dtype`.
fn policy_all_kernels_on() -> DispatchPolicy {
    DispatchPolicy {
        fused_qk: true,
        fused_qk_min_kv_seq: 0,
        sparse_attn: false,
        turbo_flash: true,
        turbo_flash_lock: false,
        turbo_flash_min_kv_seq: 0,
        planar_flash_decode: true,
        rot_k_fused: true,
    }
}

/// Prefill + `exit_prefill` + one decode step for one codec, returning the
/// decode output dtype. `exit_prefill` is what every request does, and it is
/// what leaves the bf16 K/V mirror live — the state each codec's dispatcher
/// makes its kernel-or-fallback decision in.
fn decode_output_dtype(quant: KvQuant, policy: DispatchPolicy, device: Device) -> Dtype {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut cache = KvCache::with_quant_max_seq(quant, MAX_SEQ).with_dispatch_policy(policy);

    let prefill_shape = [B, KV_H, PREFILL, HEAD_DIM];
    let prefill_k = bf16_tensor(&prefill_shape, 0x51E9_0001, device);
    let prefill_v = bf16_tensor(&prefill_shape, 0x51E9_0002, device);
    cache.enter_prefill();
    let _ = cache
        .update(&prefill_k, &prefill_v, device)
        .unwrap_or_else(|e| panic!("{quant:?}: prefill update failed: {e}"));
    cache
        .exit_prefill(device)
        .unwrap_or_else(|e| panic!("{quant:?}: exit_prefill failed: {e}"));

    let step_shape = [B, KV_H, 1, HEAD_DIM];
    let new_k = bf16_tensor(&step_shape, 0x51E9_0003, device);
    let new_v = bf16_tensor(&step_shape, 0x51E9_0004, device);
    let queries = bf16_tensor(&[B, N_Q_HEADS, 1, HEAD_DIM], 0x51E9_0005, device);

    let out = cache
        .update_and_sdpa(&queries, &new_k, &new_v, scale, "", None, device)
        .unwrap_or_else(|e| panic!("{quant:?}: decode update_and_sdpa failed: {e}"));
    out.dtype()
}

/// One reading per MSL kernel the sweep is expected to reach.
///
/// Per kernel, not per family: the aggregate counters (`fused_qk_total`,
/// `iso_flash_decode`, `rotor_flash_decode`, …) each sum two to five
/// independent kernels, and a `> 0` on a sum is satisfied by one of them.
fn read_counters() -> Vec<(&'static str, u64)> {
    vec![
        ("turbo_flash", turbo_flash_dispatch_count()),
        ("q8_fused_qk", q8_fused_qk_dispatch_count()),
        ("turbo_k3_fused_qk", turbo_k3_fused_qk_dispatch_count()),
        ("turbo_k4_fused_qk", turbo_k4_fused_qk_dispatch_count()),
        ("rotor3_fused_qk", rotor3_fused_qk_dispatch_count()),
        ("rotor4_fused_qk", rotor4_fused_qk_dispatch_count()),
        ("iso3_flash", iso3_flash_decode_dispatch_count()),
        ("iso4_flash", iso4_flash_decode_dispatch_count()),
        ("iso3_symv_flash", iso3_symv_flash_decode_dispatch_count()),
        ("iso4_symv_flash", iso4_symv_flash_decode_dispatch_count()),
        ("rotor3_flash", rotor3_flash_decode_dispatch_count()),
        ("rotor4_flash", rotor4_flash_decode_dispatch_count()),
        (
            "rotor3_symv_flash",
            rotor3_symv_flash_decode_dispatch_count(),
        ),
        (
            "rotor4_symv_flash",
            rotor4_symv_flash_decode_dispatch_count(),
        ),
    ]
}

/// Run the whole codec sweep under one policy and report every violation at
/// once — one failing codec must not hide the others.
fn sweep(policy: DispatchPolicy, arm: &str) -> Vec<String> {
    let device = Device::Gpu;
    let mut violations = Vec::new();
    for &quant in ALL_KV_QUANTS {
        let got = decode_output_dtype(quant, policy, device);
        eprintln!("{arm}: {quant:?} -> {got:?}");
        if got != ACT_DTYPE {
            violations.push(format!(
                "{quant:?}: decode returned {got:?}, want {ACT_DTYPE:?}"
            ));
        }
    }
    violations
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test kv_decode_dtype_contract -- --ignored --test-threads=1"]
fn decode_output_keeps_query_dtype_under_default_policy() {
    let violations = sweep(DispatchPolicy::default(), "default-policy");
    assert!(
        violations.is_empty(),
        "decode promoted the activation dtype under the default policy:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test kv_decode_dtype_contract -- --ignored --test-threads=1"]
fn decode_output_keeps_query_dtype_with_every_kernel_on() {
    // One reading per kernel, never a family total: `fused_qk_total` sums five
    // independent kernels, so a `> 0` on it is satisfied when only q8 fires and
    // every other codec silently took its bf16 fallback — the exact vacuity
    // this sweep exists to prevent.
    let before = read_counters();

    let violations = sweep(policy_all_kernels_on(), "all-kernels-on");

    let after = read_counters();

    assert!(
        violations.is_empty(),
        "decode promoted the activation dtype with the fused kernels on:\n  {}",
        violations.join("\n  ")
    );
    // Non-vacuity, per kernel. Every codec falls back to a bf16 path when its
    // kernel does not fire, and a bf16 path returns bf16 for free, so a zero
    // delta means the sweep measured the fallback rather than the kernel.
    //
    // Two kernels are deliberately absent from `read_counters`, both for
    // structural reasons rather than convenience:
    //
    // * `planar_flash_decode` — dormant while the bf16 K seed is live
    //   (warm-TTFT), which is the state every post-prefill decode is in, so no
    //   cache-level sweep can reach it. Pinned instead at dispatcher level by
    //   `planar_flash_decode_returns_query_dtype`.
    // * `planar_fused_qk` — same seed gate, same dispatcher.
    let mut silent = Vec::new();
    for ((name, b), (_, a)) in before.iter().zip(after.iter()) {
        let delta = a - b;
        eprintln!("all-kernels-on: {name} dispatches={delta}");
        if delta == 0 {
            silent.push(*name);
        }
    }
    assert!(
        silent.is_empty(),
        "these kernels never dispatched, so the dtype pass above says nothing \
         about them — the sweep exercised their bf16 fallbacks: {silent:?}"
    );
}
