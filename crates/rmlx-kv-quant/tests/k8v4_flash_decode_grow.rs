// K8V4 TurboFlash: decode-time growth across a power-of-two KV boundary.
//
// The K8V4 head-major "flash" decode path (`update_and_sdpa` →
// `sdpa_dispatch` → `update_and_sdpa_k8v4_flash_inner`) dispatches only once
// `kv_seq > RMLX_TURBO_FLASH_MIN`. That dispatch bypasses the legacy
// `update()` call, so it never ran `ensure_decode_capacity`: the storage
// `max_seq`, the bf16 mirror, and the latched head-major flash buffers all
// froze at the prefill window. The moment decode crossed the next power-of-two
// boundary the head-major append sliced an empty tensor off the frozen mirror
// and the step died with `reshape … size 0`.
//
// This is the same window-growth class as the rotor / planar / bf16 mirror
// decode-grow fix, but on the K8V4 flash path, which that fix did not reach.
// The growth is model-agnostic — keyed off codec + geometry (head_dim, kv_h),
// never an arch — so both real head_dim geometries (128 = Qwen3 family,
// 256 = Gemma4 family) are exercised here directly.
//
// These live in a dedicated test binary because `turbo_flash_enabled()` /
// `turbo_flash_min_kv_seq()` latch their env reads into a `OnceLock` on first
// call: the env must be set before any other code path in the process reads
// the gate. Same reasoning as `tests/apple10_head_dim_256.rs`.
//
// Run:
//   cargo test -p rmlx-kv-quant --test k8v4_flash_decode_grow -- \
//       --ignored --test-threads=1 --nocapture
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::indexing_slicing,
    unsafe_code,
    missing_docs
)]

use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

fn collect_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("materialise array on device");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect()
}

/// LCG pseudo-random data (same constants as the sibling integration tests so
/// reproducer seeds round-trip across reports).
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

/// Minimum per-row cosine between two flattened `[rows, row_len]` buffers.
fn cosine_min(a: &[f32], b: &[f32], row_len: usize) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: len mismatch");
    assert!(row_len > 0);
    let n_rows = a.len() / row_len;
    let mut mn = f32::INFINITY;
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
    }
    mn
}

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

const B: i32 = 1;
const KV_H: i32 = 2;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;

/// Reference vs grown-window agreement floor. The reference is the SAME K8V4
/// TurboFlash codec pre-sized to a window decode never crosses, so with the
/// window-grow fix the two are numerically the same up to f32 reduction-order
/// noise from the differing head-major row stride. A frozen or truncated cache
/// would blow through this floor (the attention prefix collapses).
const CODEC_FLOOR: f32 = 0.999;

fn bf16(data: &[f32], shape: &[i32], device: Device) -> Array {
    make_f32_array(data, shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 cast")
}

/// Prefill `cache` with `prefill_len` tokens of head dim `head_dim`.
fn prefill(cache: &mut KvCache, head_dim: i32, prefill_len: i32, device: Device) {
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    cache.enter_prefill();
    let shape = [B, KV_H, prefill_len, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k = bf16(&lcg_data(n, 0x1111_2222_3333_0001), &shape, device);
    let v = bf16(&lcg_data(n, 0x1111_2222_3333_0002), &shape, device);
    let q_shape = [B, N_Q_HEADS, prefill_len, head_dim];
    let qn: usize = q_shape.iter().map(|&d| d as usize).product();
    let q = bf16(&lcg_data(qn, 0x1111_2222_3333_0003), &q_shape, device);
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");
}

/// Deterministic single-token decode inputs for step `step`.
fn decode_inputs(head_dim: i32, step: u64, device: Device) -> (Array, Array, Array) {
    let one = (KV_H * head_dim) as usize;
    let k = bf16(
        &lcg_data(one, 0x5000_0000 + step),
        &[B, KV_H, 1, head_dim],
        device,
    );
    let v = bf16(
        &lcg_data(one, 0x6000_0000 + step),
        &[B, KV_H, 1, head_dim],
        device,
    );
    let q = bf16(
        &lcg_data((N_Q_HEADS * head_dim) as usize, 0x7000_0000 + step),
        &[B, N_Q_HEADS, 1, head_dim],
        device,
    );
    (k, q, v)
}

/// Drive a K8V4 TurboFlash cache and a bf16 reference cache in lock-step across
/// two power-of-two boundaries, asserting the flash decode neither crashes nor
/// diverges from bf16.
fn boundary_run(head_dim: i32, label: &str) {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // SAFETY: process-global env; --test-threads=1 is enforced by the ignore
    // annotation. Must be set before any turbo_flash_* OnceLock read.
    unsafe {
        std::env::set_var("RMLX_TURBO_FLASH", "1");
        std::env::set_var("RMLX_TURBO_FLASH_MIN", "0");
    }

    // `flash`: small window so decode crosses it cheaply. Prefill lands just
    // under the first boundary (120 < 128); decode then crosses 128 and 256.
    // `reference`: identical K8V4 TurboFlash codec, but pre-sized to a window
    // decode never crosses — so it never exercises the freeze/grow. With the
    // fix the two agree; without it, `flash` dies at the boundary.
    let max_seq_init = 128;
    let reference_window = 4096;
    let prefill_len = 120;
    let n_decode: u64 = 150; // reaches offset 270 → crosses 128 and 256

    let mut flash = KvCache::with_quant_max_seq(KvQuant::K8V4, max_seq_init);
    let mut reference = KvCache::with_quant_max_seq(KvQuant::K8V4, reference_window);
    prefill(&mut flash, head_dim, prefill_len, device);
    prefill(&mut reference, head_dim, prefill_len, device);

    let dispatch_before = turbo_flash_dispatch_count();
    let mut crossed_128 = false;
    let mut crossed_256 = false;

    for step in 0..n_decode {
        let (k, q, v) = decode_inputs(head_dim, step, device);
        let out_flash = flash
            .update_and_sdpa(&q, &k, &v, scale, "", None, device)
            .unwrap_or_else(|e| {
                panic!(
                    "{label}: flash decode step {step} (offset→{}) errored — the KV window \
                     did not grow across the boundary: {e}",
                    prefill_len as u64 + step + 1
                )
            });
        let out_ref = reference
            .update_and_sdpa(&q, &k, &v, scale, "", None, device)
            .expect("reference decode");

        let of = collect_f32(&out_flash);
        let orf = collect_f32(&out_ref);
        let cos = cosine_min(&of, &orf, head_dim as usize);
        let offset = prefill_len as u64 + step + 1;
        assert!(
            cos >= CODEC_FLOOR,
            "{label}: flash vs bf16 cosine {cos} < {CODEC_FLOOR} at offset {offset} — a frozen \
             or truncated cache changes the attention output across the boundary"
        );
        if offset >= 129 {
            crossed_128 = true;
        }
        if offset >= 257 {
            crossed_256 = true;
        }
    }

    assert_eq!(
        flash.offset(),
        prefill_len + n_decode as i32,
        "{label}: every decode step must land (offset marches with the sequence)"
    );
    assert!(crossed_128, "{label}: test must cross the 128 boundary");
    assert!(crossed_256, "{label}: test must cross the 256 boundary");
    assert!(
        turbo_flash_dispatch_count() > dispatch_before,
        "{label}: TurboFlash never dispatched — the test exercised the legacy path, \
         not the flash boundary bug"
    );
}

// ── head_dim = 128 (Qwen3 family) ───────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test k8v4_flash_decode_grow -- --ignored --test-threads=1"]
fn k8v4_flash_grows_across_pow2_boundary_head_dim_128() {
    if skip_if_no_gpu() {
        return;
    }
    boundary_run(128, "hd128");
}

// ── head_dim = 256 (Gemma4 family) ──────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test k8v4_flash_decode_grow -- --ignored --test-threads=1"]
fn k8v4_flash_grows_across_pow2_boundary_head_dim_256() {
    if skip_if_no_gpu() {
        return;
    }
    boundary_run(256, "hd256");
}
