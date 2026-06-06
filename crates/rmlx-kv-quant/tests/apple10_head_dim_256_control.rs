// Control — TurboFlash dispatch stays dormant at head_dim=256 with
// `RMLX_TURBO_FLASH=0`.
//
// Lives in a separate test binary from `apple10_head_dim_256.rs` so the
// `OnceLock<bool>`-backed `turbo_flash_enabled()` gate is exercised on the
// OFF path in its own process. Sharing a binary with the ON-path tests would
// latch `RMLX_TURBO_FLASH` to whichever test ran first lexicographically
// (Rust integration tests are alphabetised under `--test-threads=1`),
// silently zero-ing every later assertion.
//
// `#[ignore]` because it needs the Metal GPU. Run via:
//   cargo test -p rmlx-kv-quant --test apple10_head_dim_256_control -- \
//       --ignored --test-threads=1 --nocapture
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
//! head_dim=256 control: TF=0 keeps dispatch dormant.

use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
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

const B: i32 = 1;
const KV_H: i32 = 2;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;
const HEAD_DIM: i32 = 256;
const MAX_SEQ: i32 = 8192;
const SMOKE_PREFILL: i32 = 64;

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test apple10_head_dim_256_control -- --ignored --test-threads=1"]
fn turbo_flash_head_dim_256_control_dispatch_stays_dormant() {
    if skip_if_no_gpu() {
        return;
    }

    let device = Device::Gpu;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // SAFETY: process-global env var; --test-threads=1 enforced.
    unsafe {
        std::env::set_var("RMLX_TURBO_FLASH", "0");
        std::env::set_var("RMLX_TURBO_FLASH_MIN", "0");
    }

    let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ);
    cache.enter_prefill();
    let prefill_shape = [B, KV_H, SMOKE_PREFILL, HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let prefill_k = make_f32_array(&lcg_data(n_pref, 0xE3C4F5A6_B7C8_0001), &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_k");
    let prefill_v = make_f32_array(&lcg_data(n_pref, 0xE3C4F5A6_B7C8_0002), &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_v");
    let _ = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("control prefill update");
    cache.exit_prefill(device).expect("control exit_prefill");

    let dec_kv_shape = [B, KV_H, 1, HEAD_DIM];
    let q_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_dec: usize = dec_kv_shape.iter().map(|&d| d as usize).product();
    let n_q: usize = q_shape.iter().map(|&d| d as usize).product();
    let new_k = make_f32_array(&lcg_data(n_dec, 0xE3C4F5A6_B7C8_0003), &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&lcg_data(n_dec, 0xE3C4F5A6_B7C8_0004), &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");
    let q_arr = make_f32_array(&lcg_data(n_q, 0xE3C4F5A6_B7C8_0005), &q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let before = turbo_flash_dispatch_count();
    let _ = cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("control decode");
    let after = turbo_flash_dispatch_count();

    let delta = after - before;
    eprintln!("control head_dim=256 TF=0: turbo_flash_dispatch_count delta={delta}");
    assert_eq!(
        delta, 0,
        "TurboFlash dispatch must stay dormant with RMLX_TURBO_FLASH=0 at head_dim=256"
    );
}
