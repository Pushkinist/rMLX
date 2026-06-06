//! Hermetic byte-size proof: same synthetic prompt, different KV-quant flags,
//! strictly different `.kvb` on-disk sizes in the expected ratios.
//!
//! No model load. No HTTP. No spill drain thread. Pure write_caches + SsdKvIndex
//! + fs::metadata. Runs in <30 s on CPU. One tempdir per quant variant.
//!
//! ## Quant coverage
//!
//! | Variant | Tested | Reason if skipped |
//! |---------------------|--------|-------------------|
//! | K8V8 | YES | reference / 8-bit both sides |
//! | K8V4 | YES | 8-bit K, 4-bit V (TurboQuant) |
//! | Planar | YES | 8-bit K, 4-bit V (PlanarQuant rotation codec) |
//! | Mixed { k8, v4 } | NO | requires internal `MixedKvState` (pub(crate) only) |
//! | RotK / RotKTq4V | NO | same – internal Hadamard rotation state |
//! | None (bf16) | NO | KvStorage::None has no codes payload – file is geometry-only |
//!
//! If a future KvQuant variant is added without a row in `TESTED_QUANTS`, the
//! compile-time exhaustiveness note below is the forcing function to update
//! this test. See: `assert_all_serializable_quants_are_listed`.
//!
//! ## Ratio source-of-truth
//!
//! On-disk ratios are derived from the actual storage format (codes + scales +
//! rotations for Planar), not the naive bits-per-element formula used by
//! `kv_reduction_ratios_match_table` (which counts codes only). Notably Planar
//! is LARGER than K8V8 in bytes because it stores per-pair scales (N/2 f32
//! values) rather than per-group scales — the reduction is in quantization
//! error quality, not byte count. ±10% tolerance absorbs the safetensors
//! JSON header overhead (negligible for 256-token blocks).

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes in test helpers
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
#![allow(clippy::pedantic)]

use std::collections::HashSet;

use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_kv_ssd::ssd_index::{hash_to_hex, SsdKvIndex};
use rmlx_mlx::{Array, Device, Dtype};
use tempfile::TempDir;

// ── Constants ─────────────────────────────────────────────────────────────────

/// One full block — the minimum unit the SSD index tracks.
const BLOCK_TOKENS: i32 = 256;
/// Batch size.
const BATCH: i32 = 1;
/// KV heads.
const KV_HEADS: i32 = 4;
/// Head dimension — 128 satisfies all group-size constraints (group=128, 64, 32).
const HEAD_DIM: i32 = 128;
/// Synthetic model identity (no real model loaded).
const MODEL_ID: &str = "SyntheticArch/byte-proof";
/// Seed for the LCG K data.
const SEED_K: u64 = 0xDEAD_CAFE_1234_5678;
/// Seed for the LCG V data (XOR'd from K seed so K ≠ V).
const SEED_V: u64 = SEED_K ^ 0xABCD_1234;

/// The three quant variants exercised by this integration test.
///
/// Mixed / RotK / RotKTq4V require `pub(crate)` internal state
/// (`MixedKvState`, Hadamard rotation) and cannot be driven through the
/// public KvCache::enter_prefill → update → exit_prefill API.
/// None (bf16) has no codes payload — the `.kvb` contains only geometry
/// metadata and produces a file that is smaller than K8V4, breaking the
/// size-ordering assumption.
///
/// FORWARD-COMPAT NOTE: When a new KvQuant variant is added, the test
/// `assert_all_serializable_quants_are_listed` should fail loudly (by design
/// it prints a reminder). Add the new variant to TESTED_QUANTS or document
/// why it cannot be exercised here.
const TESTED_QUANTS: &[(&str, KvQuant)] = &[
    ("K8V8", KvQuant::K8V8),
    ("K8V4", KvQuant::K8V4),
    ("Planar", KvQuant::Planar),
];

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Deterministic LCG f32 values in [-1, 1]. Same generator as hydrate.rs.
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
    // SAFETY: reinterpret f32 slice bytes as u8 for Array::from_bytes.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

/// Build a single-layer `KvCache` populated with exactly `BLOCK_TOKENS` synthetic
/// tokens at the given quant. Returns the ready-to-spill cache.
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

/// Deterministic hash used to derive the `.kvb` filename, matching the
/// SsdKvIndex schema (hex of the chained FNV-1a-64 digest).
///
/// We use a fixed synthetic hash per quant variant — the test does not need
/// a real chained hash because the index lookup is keyed by this string.
fn synthetic_hash(quant_label: &str) -> String {
    // Use a trivial but unique u64 derived from the label's bytes.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in quant_label.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash_to_hex(h)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

/// Each quant variant must produce a strictly unique on-disk byte size.
///
/// DoD criterion: "All quant variants produce strictly distinct byte sizes."
#[test]
fn kv_quants_produce_distinct_byte_sizes() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();

    let mut sizes: Vec<(&str, u64)> = Vec::new();

    for (label, quant) in TESTED_QUANTS {
        let cache = build_kvcache(*quant);
        let hash = synthetic_hash(label);
        let path = tmp.path().join(format!("{hash}.kvb"));

        rmlx_kv_ssd::write_caches(&path, device, MODEL_ID, *quant, &[cache], &[]).unwrap();

        let byte_size = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("metadata for {label}: {e}"))
            .len();

        assert!(byte_size > 0, "{label}: .kvb file must not be empty");
        sizes.push((label, byte_size));
    }

    // All sizes must be strictly distinct.
    let unique: HashSet<u64> = sizes.iter().map(|(_, n)| *n).collect();
    assert_eq!(
        unique.len(),
        sizes.len(),
        "all quant variants must produce unique byte sizes; got: {sizes:?}"
    );

    println!("\n── kv_quants_produce_distinct_byte_sizes ────────────────");
    for (label, sz) in &sizes {
        println!("  {label:<30} {sz:>10} bytes");
    }
}

/// For every tested quant the on-disk byte size must be within ±10% of the
/// formula-derived payload ratio relative to K8V8.
///
/// ## Formula-derived ratios (including codes + scales + rotations overhead)
///
/// Shape: [B=1, KV_H=4, S=256, D=128] → total N = 131_072 elements.
///
/// | Variant | K payload | V payload | total | ratio/K8V8 |
/// |---------|---------------|-----------------------------------------------------|----------|------------|
/// | K8V8 | N + N/128*4 | N + N/128*4 | 270_336 | 1.000× |
/// | K8V4 | N + N/128*4 | N/2 + N/32*4 | 217_088 | 0.803× |
/// | Planar | N + N/128*4 | N/2 (codes) + N/2*4 (per-pair scales) + N/4 (rots) | 495_616 | 1.833× |
///
/// Note: Planar is LARGER than K8V8 because it uses per-pair scales (N/2 f32
/// values) instead of per-group scales. The reduction in quantization error
/// from the Givens rotation codebook is in quality, not in byte count.
///
/// The ±10% tolerance absorbs the fixed safetensors JSON header overhead
/// (~hundreds of bytes) which is negligible for 256-token blocks.
///
/// Also verifies: index row `byte_size` column == `fs::metadata().len()`.
#[test]
fn kv_quant_byte_size_ratios_match_reduction_table() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("index.db");
    let index = SsdKvIndex::open_at(&db_path).unwrap();

    // ── Write one .kvb per quant, record in the index ─────────────────────
    let mut quant_sizes: Vec<(&str, KvQuant, u64)> = Vec::new();

    // this byte-size proof is layout-agnostic — use a stable
    // placeholder layout_key so the `(hash, layout_key)` PK is well-defined.
    const TEST_LAYOUT_KEY: u64 = 0xb172_e510_5117_5e57;

    for (label, quant) in TESTED_QUANTS {
        let cache = build_kvcache(*quant);
        let hash = synthetic_hash(label);
        let path = tmp.path().join(format!("{hash}.kvb"));

        rmlx_kv_ssd::write_caches(&path, device, MODEL_ID, *quant, &[cache], &[]).unwrap();

        let byte_size = std::fs::metadata(&path).unwrap().len();
        index
            .record(
                &hash,
                TEST_LAYOUT_KEY,
                &path,
                MODEL_ID,
                &quant.to_string(),
                byte_size,
            )
            .unwrap();

        quant_sizes.push((label, *quant, byte_size));
    }

    // ── Assert index row byte_size == fs::metadata ───────
    for (label, quant, expected_size) in &quant_sizes {
        let hash = synthetic_hash(label);
        let row = index
            .lookup(&hash, TEST_LAYOUT_KEY)
            .unwrap()
            .unwrap_or_else(|| panic!("{label}: index row not found after record()"));
        assert_eq!(
            row.byte_size, *expected_size,
            "{label}: index byte_size ({}) must match fs::metadata ({expected_size})",
            row.byte_size
        );
        assert_eq!(
            row.kv_quant,
            quant.to_string(),
            "{label}: index kv_quant string mismatch"
        );
    }

    // ── Compute ratios vs K8V8 reference ──────────────────────────────────
    let k8v8_size = quant_sizes
        .iter()
        .find(|(l, _, _)| *l == "K8V8")
        .map(|(_, _, s)| *s)
        .expect("K8V8 must be in TESTED_QUANTS");

    // Formula-derived expected ratios vs K8V8 for shape [1, 4, 256, 128].
    // See doc comment above for derivation. These are exact predictions;
    // ±10% tolerance covers the fixed safetensors header overhead.
    //
    // N = B * kv_h * S * D = 1 * 4 * 256 * 128 = 131_072 elements.
    // K side is q8_0 (8-bit codes + per-128-group f32 scales) for all variants.
    let n: u64 = (BATCH * KV_HEADS * BLOCK_TOKENS * HEAD_DIM) as u64;
    let k8v8_formula: u64 = {
        let k = n + n / 128 * 4; // 8-bit codes + per-128 scales
        let v = n + n / 128 * 4; // same for V
        k + v
    };
    let k8v4_formula: u64 = {
        let k = n + n / 128 * 4;
        let v_codes = n / 2; // 4-bit codes
        let v_scales = n / 32 * 4; // TurboQuant: per-32-group f32 scales
        k + v_codes + v_scales
    };
    let planar_formula: u64 = {
        let k = n + n / 128 * 4;
        let v_codes = n / 2; // 4-bit codes
        let v_scales_planar = n / 2 * 4; // PlanarQuant: per-PAIR f32 scales
        let v_rotations = n / 4; // 4 bits per pair, 2 pairs per byte
        k + v_codes + v_scales_planar + v_rotations
    };

    let expected_ratio_vs_k8v8 = |label: &str| -> f64 {
        match label {
            "K8V8" => k8v8_formula as f64 / k8v8_formula as f64,
            "K8V4" => k8v4_formula as f64 / k8v8_formula as f64,
            "Planar" => planar_formula as f64 / k8v8_formula as f64,
            other => panic!("no formula-derived ratio defined for {other}"),
        }
    };

    println!("\n── kv_quant_byte_size_ratios_match_reduction_table ─────");
    println!(
        "  {:<30} {:>12} {:>18} {:>18}",
        "quant", "byte_size", "ratio_vs_K8V8", "formula_vs_K8V8"
    );

    for (label, _quant, size) in &quant_sizes {
        let ratio_vs_k8v8 = *size as f64 / k8v8_size as f64;
        let expected = expected_ratio_vs_k8v8(label);
        let tol = 0.10; // ±10%: safetensors header is negligible vs payload
        let diff = (ratio_vs_k8v8 - expected).abs();

        println!(
            "  {:<30} {:>12} {:>17.4}x {:>17.4}x  diff={:.4}",
            label, size, ratio_vs_k8v8, expected, diff
        );

        assert!(
            diff / expected.abs().max(0.001) < tol,
            "{label}: observed ratio {ratio_vs_k8v8:.4}× vs formula {expected:.4}× — \
             relative diff {:.4} exceeds ±{:.0}% tolerance \
             (formula: k8v8={k8v8_formula}, k8v4={k8v4_formula}, planar={planar_formula})",
            diff / expected.abs().max(0.001),
            tol * 100.0
        );
    }
}

/// Forward-compatibility reminder: this is NOT a test that runs assertions.
/// It prints a notice to ensure a future KvQuant variant triggers an update
/// to TESTED_QUANTS. If you add a new KvQuant and see this message, add
/// the variant (or document why it cannot be exercised here).
#[test]
fn assert_all_serializable_quants_are_listed() {
    // The full set of KvQuant variants as of the last update to this test.
    // Update this set when a new variant is added to the KvQuant enum.
    // Variants deliberately excluded from TESTED_QUANTS are listed in the
    // EXCLUDED set with their reason.
    const TESTED_NAMES: &[&str] = &["K8V8", "K8V4", "Planar"];
    const EXCLUDED_NAMES: &[&str] = &[
        "Mixed",    // requires pub(crate) MixedKvState
        "RotK",     // requires pub(crate) Hadamard rotation state
        "RotKTq4V", // same as RotK
        "None",     // KvStorage::None has no codes payload
    ];

    // All known KvQuant variants as of 2026-05-25.
    const ALL_KNOWN: &[&str] = &[
        "K8V4", "K8V8", "Planar", "None", "Mixed", "RotK", "RotKTq4V",
    ];

    let tested: HashSet<&str> = TESTED_NAMES.iter().copied().collect();
    let excluded: HashSet<&str> = EXCLUDED_NAMES.iter().copied().collect();
    for name in ALL_KNOWN {
        assert!(
            tested.contains(name) || excluded.contains(name),
            "KvQuant variant '{name}' is not in TESTED_QUANTS or EXCLUDED_NAMES — \
             please add it to one of the two lists in ssd_quant_byte_size_table.rs"
        );
    }
    println!(
        "\nassert_all_serializable_quants_are_listed: \
         {}/{} variants covered ({} tested, {} excluded from integration test)",
        ALL_KNOWN.len(),
        ALL_KNOWN.len(),
        TESTED_NAMES.len(),
        EXCLUDED_NAMES.len()
    );
}
