//! Criterion micro-benchmarks for `rmlx-quant` dequant kernels.
//!
//! Covers per-element decoders and the outer row-iter dequant functions for
//! affine and mxfp families, plus the TurboQuant and PlanarQuant outer
//! dequantize paths.
//!
//! # Running
//!
//! ```
//! cargo bench -p rmlx-quant --bench dequant
//! ```
//!
//! # Design choices (per perf-book ch 2)
//!
//! - `std::hint::black_box` on both inputs and outputs to prevent dead-code
//!   elimination.
//! - `Throughput::Elements` so Criterion reports GB/s–equivalent tok/s numbers.
//! - One bench per kernel family; a group per representative shape.
//! - NOT wired into `make ci` — micro-bench noise must not gate the CI (the
//!   perf-book explicitly warns against this).
//! - Fixture construction is outside the timed loop (in `b.iter_batched` setup
//!   or in the group body before `b.bench_function`).

#![allow(
    missing_docs, // criterion_group!/criterion_main! expand to undocumented fns
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    // disallowed_methods is a separate lint from unwrap_used;
    // benchmark code (bucket-B equivalent) is already exempted for unwrap_used.
    clippy::disallowed_methods,
)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rmlx_kv_quant::{
    planarquant::{planar_dequantize, planar_quantize},
    turboquant::{turbo_dequantize, turbo_quantize_v},
};
use rmlx_quant::{
    affine::{dequant_to_f32 as affine_dequant, AffineParams, CodeStorage},
    bf16::bf16_decode_into,
    fp4::e2m1_decode,
    fp8::{e4m3_decode, e8m0_decode, ue4m3_decode},
    mxfp::{dequant_to_f32 as mxfp_dequant, MxFamily, MxParams},
};

// ── Per-element scalar decoders ───────────────────────────────────────────────

/// Representative batch size for per-element decoder benches.
/// 1 MiB / 1 byte per element = 1M elements — enough to amortise function-call
/// overhead while fitting comfortably in L2.
const SCALAR_N: usize = 1 << 20; // 1 048 576 elements

fn bench_bf16_to_f32(c: &mut Criterion) {
    // Build a synthetic bf16 LE byte array: alternating non-zero patterns.
    let data: Vec<u8> = (0..SCALAR_N * 2)
        .map(|i| (i as u8).wrapping_add(1))
        .collect();
    let mut out = vec![0.0f32; SCALAR_N];

    let mut g = c.benchmark_group("per_element");
    g.throughput(Throughput::Elements(SCALAR_N as u64));
    g.bench_function("bf16_decode_into", |b| {
        b.iter(|| {
            bf16_decode_into(black_box(&data), black_box(&mut out)).unwrap();
        });
    });
    g.finish();
}

fn bench_fp8_decoders(c: &mut Criterion) {
    // Build 256-entry lookup tables (all possible byte values) to force the
    // compiler to materialise the result rather than constant-fold.
    let bytes: Vec<u8> = (0..=255).collect();

    let mut g = c.benchmark_group("per_element");
    // Throughput in terms of the 256-element table (the loop inside the bench
    // iterates the table repeatedly).
    g.throughput(Throughput::Elements(256));

    g.bench_function("e8m0_decode_256", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &byte in black_box(&bytes) {
                acc += e8m0_decode(black_box(byte));
            }
            black_box(acc);
        });
    });
    g.bench_function("e4m3_decode_256", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &byte in black_box(&bytes) {
                acc += e4m3_decode(black_box(byte));
            }
            black_box(acc);
        });
    });
    g.bench_function("ue4m3_decode_256", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &byte in black_box(&bytes) {
                acc += ue4m3_decode(black_box(byte));
            }
            black_box(acc);
        });
    });

    g.finish();
}

fn bench_e2m1_decode(c: &mut Criterion) {
    // fp4 nibbles are 0..15 (4-bit values).
    let nibbles: Vec<u8> = (0..16).collect();

    let mut g = c.benchmark_group("per_element");
    g.throughput(Throughput::Elements(16));
    g.bench_function("e2m1_decode_16", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for &nib in black_box(&nibbles) {
                acc += e2m1_decode(black_box(nib));
            }
            black_box(acc);
        });
    });
    g.finish();
}

// ── Affine dequant (row-iter outer kernel) ────────────────────────────────────

/// Build an affine 4-bit U32Le fixture: rows × cols, all codes = midpoint,
/// scale = 1.0, bias = 0.0.
///
/// All-zero packed bytes produce code=0 for every element — valid and cheap to
/// construct.
fn affine_fixture(bits: u8, group_size: u32, rows: usize, cols: usize) -> AffineFixture {
    let params = AffineParams {
        bits,
        group_size,
        storage: CodeStorage::U32Le,
        rows,
        cols,
    };

    // U32Le storage: ceil(cols / (32/bits)) u32 words per row.
    let per_word = 32 / bits as usize;
    let words_per_row = cols.div_ceil(per_word);
    let packed_len = rows * words_per_row * 4; // 4 bytes per u32
    let packed = vec![0u8; packed_len];

    // Scale buffer: rows × (cols/group_size) groups, each 2 bytes bf16.
    // bf16(1.0) = 0x3F80 → LE bytes [0x80, 0x3F].
    let groups_per_row = cols / group_size as usize;
    let sb_len = rows * groups_per_row * 2;
    let mut scales = vec![0u8; sb_len];
    let mut biases = vec![0u8; sb_len];
    for i in (0..sb_len).step_by(2) {
        scales[i] = 0x80;
        scales[i + 1] = 0x3F; // bf16(1.0)
        biases[i] = 0x00;
        biases[i + 1] = 0x00; // bf16(0.0)
    }

    let out = vec![0.0f32; rows * cols];
    AffineFixture {
        params,
        packed,
        scales,
        biases,
        out,
    }
}

struct AffineFixture {
    params: AffineParams,
    packed: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
    out: Vec<f32>,
}

fn bench_affine_dequant(c: &mut Criterion) {
    // Two representative shapes from Gemma4-e4b-it-mxfp8 affine weight layers:
    // (a) q4 small — a typical GQA head projection: 256×2560, g64
    // (b) q4 large — attention projection: 2560×2560, g64
    // Both use 4-bit U32Le (standard MLX affine format).
    struct Shape {
        label: &'static str,
        bits: u8,
        group_size: u32,
        rows: usize,
        cols: usize,
    }
    let shapes = [
        Shape {
            label: "q4_g64_256x2560",
            bits: 4,
            group_size: 64,
            rows: 256,
            cols: 2560,
        },
        Shape {
            label: "q4_g64_2560x2560",
            bits: 4,
            group_size: 64,
            rows: 2560,
            cols: 2560,
        },
    ];

    let mut g = c.benchmark_group("affine_dequant");
    for s in &shapes {
        let n_elements = (s.rows * s.cols) as u64;
        g.throughput(Throughput::Elements(n_elements));
        let mut fix = affine_fixture(s.bits, s.group_size, s.rows, s.cols);
        g.bench_function(s.label, |b| {
            b.iter(|| {
                affine_dequant(
                    black_box(&fix.params),
                    black_box(&fix.packed),
                    black_box(&fix.scales),
                    black_box(&fix.biases),
                    black_box(&mut fix.out),
                )
                .unwrap();
            });
        });
    }
    g.finish();
}

// ── mxfp dequant (row-iter outer kernel) ─────────────────────────────────────

struct MxFixture {
    params: MxParams,
    packed: Vec<u8>,
    scales: Vec<u8>,
    out: Vec<f32>,
}

fn mxfp8_fixture(rows: usize, cols: usize) -> MxFixture {
    let params = MxParams {
        family: MxFamily::Mxfp8,
        rows,
        cols,
    };
    let gs = 32usize;
    let groups_per_row = cols / gs;
    // E8M0 scale 0x3F = 2^(-64) → small but non-NaN.
    let scales = vec![0x3Fu8; rows * groups_per_row];
    // E4M3 elements: use 0x3C = 1.0 for all.
    let packed = vec![0x3Cu8; rows * cols];
    let out = vec![0.0f32; rows * cols];
    MxFixture {
        params,
        packed,
        scales,
        out,
    }
}

fn bench_mxfp_dequant(c: &mut Criterion) {
    // Shape matches primary test model (Gemma4-e4b-it-mxfp8) attention projection.
    // mxfp8 g32, shape 2560×2560 is representative for the KV-path projection.
    struct Shape {
        label: &'static str,
        rows: usize,
        cols: usize,
    }
    let shapes = [
        Shape {
            label: "mxfp8_g32_256x2560",
            rows: 256,
            cols: 2560,
        },
        Shape {
            label: "mxfp8_g32_2560x2560",
            rows: 2560,
            cols: 2560,
        },
    ];

    let mut g = c.benchmark_group("mxfp_dequant");
    for s in &shapes {
        let n_elements = (s.rows * s.cols) as u64;
        g.throughput(Throughput::Elements(n_elements));
        let mut fix = mxfp8_fixture(s.rows, s.cols);
        g.bench_function(s.label, |b| {
            b.iter(|| {
                mxfp_dequant(
                    black_box(&fix.params),
                    black_box(&fix.packed),
                    black_box(&fix.scales),
                    black_box(&mut fix.out),
                )
                .unwrap();
            });
        });
    }
    g.finish();
}

// ── TurboQuant dequant ────────────────────────────────────────────────────────

fn bench_turbo_dequant(c: &mut Criterion) {
    // KV-cache shape: [1, 2, 512, 256] — Gemma4-e2b GQA 2 KV heads, 512 seq,
    // 256 head-dim (2×4-bit = typical V4 path).
    let shape = [1i32, 2, 512, 256];
    let n_elements = shape.iter().map(|&x| x as u64).product::<u64>();

    // Build random-ish f32 data (not truly random — just a deterministic pattern
    // that spans the codebook range).
    let data: Vec<f32> = (0..n_elements as usize)
        .map(|i| ((i % 128) as f32 - 64.0) / 64.0)
        .collect();

    let mut g = c.benchmark_group("turboquant_dequant");
    g.throughput(Throughput::Elements(n_elements));

    // Pre-quantize once outside the timed loop.
    let blocks_4bit = turbo_quantize_v(&data, 4, &shape).unwrap();
    let blocks_2bit = turbo_quantize_v(&data, 2, &shape).unwrap();

    g.bench_function("turbo_dequant_4bit_1x2x512x256", |b| {
        b.iter(|| {
            let _ = black_box(turbo_dequantize(black_box(&blocks_4bit)).unwrap());
        });
    });

    g.bench_function("turbo_dequant_2bit_1x2x512x256", |b| {
        b.iter(|| {
            let _ = black_box(turbo_dequantize(black_box(&blocks_2bit)).unwrap());
        });
    });

    g.finish();
}

// ── PlanarQuant dequant ───────────────────────────────────────────────────────

fn bench_planar_dequant(c: &mut Criterion) {
    // KV-cache shape: [1, 2, 512, 256] — same as TurboQuant bench for apples-to-
    // apples comparison. PlanarQuant requires D (last dim) multiple of 32.
    let shape = [1i32, 2, 512, 256];
    let n_elements = shape.iter().map(|&x| x as u64).product::<u64>();

    let data: Vec<f32> = (0..n_elements as usize)
        .map(|i| ((i % 128) as f32 - 64.0) / 64.0)
        .collect();

    let mut g = c.benchmark_group("planarquant_dequant");
    g.throughput(Throughput::Elements(n_elements));

    let blocks_4bit = planar_quantize(&data, 32, 4, &shape).unwrap();
    let blocks_2bit = planar_quantize(&data, 32, 2, &shape).unwrap();

    g.bench_function("planar_dequant_4bit_1x2x512x256", |b| {
        b.iter(|| {
            let _ = black_box(planar_dequantize(black_box(&blocks_4bit)).unwrap());
        });
    });

    g.bench_function("planar_dequant_2bit_1x2x512x256", |b| {
        b.iter(|| {
            let _ = black_box(planar_dequantize(black_box(&blocks_2bit)).unwrap());
        });
    });

    g.finish();
}

// ── Criterion entry point ─────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_bf16_to_f32,
    bench_fp8_decoders,
    bench_e2m1_decode,
    bench_affine_dequant,
    bench_mxfp_dequant,
    bench_turbo_dequant,
    bench_planar_dequant,
);
criterion_main!(benches);
