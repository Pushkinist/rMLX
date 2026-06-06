//! CLI table printer: on-disk byte sizes for every tested KV-quant variant.
//!
//! No model load. No HTTP. Writes one `.kvb` per quant to a tempdir, stats it,
//! then prints a summary table.
//!
//! ```text
//! cargo run --release -p rmlx-models --example ssd_quant_byte_size
//! ```
//!
//! Output format:
//!
//! ```text
//! quant | byte_size | ratio_vs_K8V8 | description
//! K8V8 | 270832 | 1.000x | 8-bit K + 8-bit V (affine q8_0)
//! K8V4 | 217584 | 0.803x | 8-bit K + 4-bit V (TurboQuant)
//! Planar | 496200 | 1.832x | 8-bit K + Planar-4-bit V (per-pair scales+rot)
//! ```
//!
//! Shape used: [B=1, KV_H=4, S=256, D=128] — one full 256-token block.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes in example helpers
#![allow(unsafe_code)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};
use tempfile::TempDir;

const BLOCK_TOKENS: i32 = 256;
const BATCH: i32 = 1;
const KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 128;
const MODEL_ID: &str = "SyntheticArch/byte-proof";
const SEED_K: u64 = 0xDEAD_CAFE_1234_5678;
const SEED_V: u64 = SEED_K ^ 0xABCD_1234;

/// Quants to measure. Extend when new KvQuant variants support the public
/// enter_prefill → update → exit_prefill API.
const QUANTS: &[(&str, KvQuant, &str)] = &[
    ("K8V8", KvQuant::K8V8, "8-bit K + 8-bit V (affine q8_0)"),
    ("K8V4", KvQuant::K8V4, "8-bit K + 4-bit V (TurboQuant)"),
    (
        "Planar",
        KvQuant::Planar,
        "8-bit K + Planar-4-bit V (per-pair scales+rot)",
    ),
];

fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) as f32 / u32::MAX as f32).mul_add(2.0, -1.0)
        })
        .collect()
}

fn arr(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

fn build_kvcache(quant: KvQuant) -> KvCache {
    let device = Device::Cpu;
    let shape = [BATCH, KV_HEADS, BLOCK_TOKENS, HEAD_DIM];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = arr(&lcg(n, SEED_K), &shape);
    let v = arr(&lcg(n, SEED_V), &shape);
    let mut c = KvCache::with_quant_max_seq(quant, 4096);
    c.enter_prefill();
    c.update(&k, &v, device).unwrap();
    c.exit_prefill(device).unwrap();
    c
}

fn main() {
    let device = Device::Cpu;
    let tmp = TempDir::new().expect("tempdir");

    let mut rows: Vec<(&str, u64, &str)> = Vec::new();

    for (label, quant, desc) in QUANTS {
        let cache = build_kvcache(*quant);
        let path = tmp.path().join(format!("{label}.kvb"));
        rmlx_kv_ssd::write_caches(&path, device, MODEL_ID, *quant, &[cache], &[])
            .unwrap_or_else(|e| panic!("write_caches failed for {label}: {e}"));
        let byte_size = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("metadata failed for {label}: {e}"))
            .len();
        tracing::debug!(quant = label, byte_size, "measured");
        rows.push((label, byte_size, desc));
    }

    let k8v8_size = rows
        .iter()
        .find(|(l, _, _)| *l == "K8V8")
        .map_or(1, |(_, s, _)| *s);

    println!(
        "\n{:<12} | {:>12} | {:>15} | description",
        "quant", "byte_size", "ratio_vs_K8V8"
    );
    println!("{}", "-".repeat(80));
    for (label, size, desc) in &rows {
        let ratio = *size as f64 / k8v8_size as f64;
        println!("{label:<12} | {size:>12} | {ratio:>14.3}x | {desc}");
    }
    println!();
    println!(
        "Shape: [B={BATCH}, KV_H={KV_HEADS}, S={BLOCK_TOKENS}, D={HEAD_DIM}] \
         = {} elements/layer",
        BATCH * KV_HEADS * BLOCK_TOKENS * HEAD_DIM
    );
    println!("Model ID: {MODEL_ID} (synthetic, no real model loaded)");
}
