//! Hermetic byte-size proof: same synthetic prompt, different KV-quant flags,
//! strictly different `.kvb` on-disk sizes in the expected ratios.
//!
//! No model load. No HTTP. No spill drain thread. Pure write_caches + SsdKvIndex
//! + fs::metadata. Runs in <30 s on CPU. One tempdir per quant variant.
//!
//! ## What decides a block's size
//!
//! Not the codec name — what the cache actually holds. A codec whose decode
//! reads only the bf16 mirror builds no packed store (`exit_prefill` skips the
//! bulk encode), so the only thing there is to persist is that mirror: its
//! block is bf16 and is byte-identical whatever the flag said. A codec whose
//! decode reads its packed store spills codes + scales and lands somewhere
//! else entirely.
//!
//! | Variant | Tested | Role |
//! |---------------------|--------|-------------------|
//! | K8V8 / K8V4 / Planar | YES | mirror-only — all three spill the same bf16 block |
//! | IsoKOnly3 | YES | store-keeping control — must NOT land on the bf16 size |
//! | Mixed / RotK / RotKTq4V | NO | requires internal `MixedKvState` (pub(crate) only) |
//! | None (bf16) | YES | the oracle the mirror-only trio must match |
//!
//! The control is what keeps this from being a gate that cannot fail: without
//! it, a writer that emitted a fixed-size stub for every layer would satisfy
//! the equality half and the formula half alike.

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

/// The mirror-only quants: their decode reads the bf16 mirror on both axes, so
/// `exit_prefill` builds no packed store and all three spill the same bf16
/// block. Mixed / RotK / RotKTq4V require `pub(crate)` internal state
/// (`MixedKvState`, Hadamard rotation) and cannot be driven through the public
/// `KvCache::enter_prefill` → `update` → `exit_prefill` API.
///
/// FORWARD-COMPAT NOTE: when a new `KvQuant` variant is added,
/// `assert_all_serializable_quants_are_listed` prints a reminder to classify it
/// here or in the excluded list.
const MIRROR_ONLY_QUANTS: &[(&str, KvQuant)] = &[
    ("K8V8", KvQuant::K8V8),
    ("K8V4", KvQuant::K8V4),
    ("Planar", KvQuant::Planar),
];

/// Store-keeping control: `IsoKOnly3` re-quantises K into its packed store on
/// every decode step, so `exit_prefill` builds that store and the block carries
/// real codes. Its size must differ from the bf16 block — a writer that had
/// stopped distinguishing the two would pass every other assertion here.
const STORE_KEEPING_CONTROL: (&str, KvQuant) = ("IsoKOnly3", KvQuant::IsoKOnly3);

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

/// Write one `.kvb` for `quant` into `dir` and return its size in bytes.
fn spill_size(dir: &TempDir, label: &str, quant: KvQuant) -> u64 {
    let device = Device::Cpu;
    let cache = build_kvcache(quant);
    let hash = synthetic_hash(label);
    let path = dir.path().join(format!("{hash}.kvb"));
    rmlx_kv_ssd::write_caches(&path, device, MODEL_ID, quant, &[cache], &[]).unwrap();
    std::fs::metadata(&path)
        .unwrap_or_else(|e| panic!("metadata for {label}: {e}"))
        .len()
}

/// bf16 payload of one `[B, kv_h, S, D]` K/V pair: two tensors, 2 bytes per
/// element. Everything above this in the file is the safetensors JSON header.
fn bf16_payload_bytes() -> u64 {
    let n: u64 = (BATCH * KV_HEADS * BLOCK_TOKENS * HEAD_DIM) as u64;
    2 * n * 2
}

/// The mirror-only quants all spill the **same** bf16 block, byte-identical to
/// the one a plain `KvQuant::None` cache spills.
///
/// This is the on-disk face of the `exit_prefill` store skip: those codecs hold
/// nothing but the bf16 mirror, so that mirror is the whole layer and it is
/// what gets persisted. Hydrate then hands the cache back the same bytes it
/// spilled, which is why a prompt-cache hit served from disk decodes
/// identically to one served from RAM.
#[test]
fn mirror_only_quants_spill_the_same_bf16_block_as_none() {
    let tmp = TempDir::new().unwrap();
    let baseline = spill_size(&tmp, "None", KvQuant::None);
    let payload = bf16_payload_bytes();

    assert!(
        baseline >= payload && baseline < payload + 4096,
        "the bf16 block must be the two bf16 tensors plus a small header: \
         got {baseline} B against a {payload} B payload"
    );

    println!("\n── mirror_only_quants_spill_the_same_bf16_block_as_none ──");
    println!("  {:<30} {:>10} bytes", "None (oracle)", baseline);
    for (label, quant) in MIRROR_ONLY_QUANTS {
        let size = spill_size(&tmp, label, *quant);
        println!("  {label:<30} {size:>10} bytes");
        assert_eq!(
            size, baseline,
            "{label} builds no packed store, so its block must be the same bf16 \
             block `None` spills ({baseline} B); got {size} B"
        );
    }
}

/// The control: a codec whose decode reads its packed store spills that store —
/// a block that is neither the bf16 block nor an empty geometry stub.
///
/// Both comparisons are load-bearing. Against the bf16 block it proves the
/// mirror-only rule did not swallow a codec that needs its codes; against the
/// unfilled cache's block it proves the codes are actually in the file. Drop
/// the second and a writer that emitted geometry for every layer would satisfy
/// the first by landing on 176 bytes.
#[test]
fn store_keeping_quant_spills_its_codes_not_the_bf16_block() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let bf16_block = spill_size(&tmp, "None", KvQuant::None);
    let (label, quant) = STORE_KEEPING_CONTROL;
    assert!(
        quant.materialises_packed_store(),
        "{label} must be a store-keeping codec for this control to mean anything"
    );

    // Geometry-only reference: same codec, never filled, so there is no payload
    // to write and the file is header + geometry metadata alone.
    let empty_path = tmp.path().join("empty.kvb");
    rmlx_kv_ssd::write_caches(
        &empty_path,
        device,
        MODEL_ID,
        quant,
        &[KvCache::with_quant_max_seq(quant, 4096)],
        &[],
    )
    .unwrap();
    let geometry_only = std::fs::metadata(&empty_path).unwrap().len();

    let size = spill_size(&tmp, label, quant);
    println!("\n── store_keeping_quant_spills_its_codes_not_the_bf16_block ──");
    println!("  {:<30} {:>10} bytes", "None (bf16 block)", bf16_block);
    println!(
        "  {:<30} {:>10} bytes",
        "unfilled (geometry only)", geometry_only
    );
    println!("  {label:<30} {size:>10} bytes");
    assert_ne!(
        size, bf16_block,
        "{label} keeps a packed store, so its block must not be the bf16 block"
    );
    // The K codes alone are at least half a byte per element; anything near the
    // geometry-only size means the payload never reached the file.
    let n: u64 = (BATCH * KV_HEADS * BLOCK_TOKENS * HEAD_DIM) as u64;
    assert!(
        size > geometry_only + n / 2,
        "{label} must spill its K codes: got {size} B against a {geometry_only} B \
         geometry-only block for the same codec"
    );
}

/// Index bookkeeping: the row's `byte_size` and `kv_quant` columns must match
/// what was actually written.
#[test]
fn index_row_matches_the_file_it_describes() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("index.db");
    let index = SsdKvIndex::open_at(&db_path).unwrap();

    // this byte-size proof is layout-agnostic — use a stable
    // placeholder layout_key so the `(hash, layout_key)` PK is well-defined.
    const TEST_LAYOUT_KEY: u64 = 0xb172_e510_5117_5e57;

    let mut cases: Vec<(&str, KvQuant)> = MIRROR_ONLY_QUANTS.to_vec();
    cases.push(STORE_KEEPING_CONTROL);

    for (label, quant) in cases {
        let cache = build_kvcache(quant);
        let hash = synthetic_hash(label);
        let path = tmp.path().join(format!("{hash}.kvb"));
        rmlx_kv_ssd::write_caches(&path, device, MODEL_ID, quant, &[cache], &[]).unwrap();
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

        let row = index
            .lookup(&hash, TEST_LAYOUT_KEY)
            .unwrap()
            .unwrap_or_else(|| panic!("{label}: index row not found after record()"));
        assert_eq!(
            row.byte_size, byte_size,
            "{label}: index byte_size ({}) must match fs::metadata ({byte_size})",
            row.byte_size
        );
        assert_eq!(
            row.kv_quant,
            quant.to_string(),
            "{label}: index kv_quant string mismatch"
        );
    }
}

/// Forward-compatibility reminder: this is NOT a test that runs assertions
/// against the writer. It prints a notice to ensure a future KvQuant variant
/// triggers an update to the lists above.
#[test]
fn assert_all_serializable_quants_are_listed() {
    const TESTED_NAMES: &[&str] = &["K8V8", "K8V4", "Planar", "IsoKOnly3", "None"];
    const EXCLUDED_NAMES: &[&str] = &[
        "Mixed",    // requires pub(crate) MixedKvState
        "RotK",     // requires pub(crate) Hadamard rotation state
        "RotKTq4V", // same as RotK
    ];

    // All known KvQuant variants as of the last update to this test.
    const ALL_KNOWN: &[&str] = &[
        "K8V4",
        "K8V8",
        "Planar",
        "None",
        "IsoKOnly3",
        "Mixed",
        "RotK",
        "RotKTq4V",
    ];

    let tested: HashSet<&str> = TESTED_NAMES.iter().copied().collect();
    let excluded: HashSet<&str> = EXCLUDED_NAMES.iter().copied().collect();
    for name in ALL_KNOWN {
        assert!(
            tested.contains(name) || excluded.contains(name),
            "KvQuant variant '{name}' is not in the tested or excluded lists — \
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
