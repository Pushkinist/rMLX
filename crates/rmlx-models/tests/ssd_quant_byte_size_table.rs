//! Hermetic byte-size proof: same synthetic prompt, every KV-quant flag, and
//! the assertion that a block's payload is decided by what the cache holds
//! rather than by which codec was asked for.
//!
//! No model load. No HTTP. No spill drain thread. Pure write_caches + SsdKvIndex
//! + safetensors payload accounting. Runs in ~25 s on CPU.
//!
//! ## What decides a block's size
//!
//! Not the codec name — what the cache actually holds. A codec whose decode
//! reads only the bf16 mirror builds no packed store (`exit_prefill` skips the
//! bulk encode), so the only thing there is to persist is that mirror: its
//! block is bf16 and its payload is byte-identical whatever the flag said. A
//! codec whose decode reads its packed store spills codes + scales and lands
//! somewhere else entirely.
//!
//! | Group | Source | Role |
//! |---|---|---|
//! | mirror-only | derived: `!materialises_packed_store()` | every one must spill the same bf16 payload |
//! | store-keeping | derived: `materialises_packed_store()` | must spill codes, not the bf16 payload and not an empty stub |
//! | `Mixed` / `RotK` | `UNDRIVABLE` | needs `pub(crate)` `MixedKvState`; refuses `update()` by contract |
//! | `None` (bf16) | the oracle | what the mirror-only group must match |
//!
//! Two things keep this from being a gate that cannot fail. The groups are
//! **derived from the predicate**, not listed, so a codec added to the enum is
//! swept on the next run rather than silently skipped; and the store-keeping
//! group is compared against the *same codec's* unfilled block, so a writer
//! that emitted geometry for every layer would fail rather than satisfy an
//! equality by collapsing everything onto one number.
//!
//! Sizes are **payloads**, not file lengths: the safetensors header carries
//! `kv_quant.to_string()`, so whole-file equality would turn a codec rename
//! into a failure that reads as "the packed store came back".

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

/// Codecs this integration test cannot drive, with the reason.
///
/// `Mixed` and `RotK` need `pub(crate)` internal state
/// (`MixedKvState`, the Hadamard rotation) that the public
/// `KvCache::enter_prefill` → `update` → `exit_prefill` API cannot populate;
/// they also refuse `update()` by contract.
///
/// Everything else is classified at runtime off
/// `KvQuant::materialises_packed_store()`, not off a list kept here — that is
/// what keeps the coverage check from drifting behind the enum.
const UNDRIVABLE: &[(&str, &str)] = &[
    ("mixed_k8g64_v4g64", "requires pub(crate) MixedKvState"),
    ("rot_k_v8g64", "requires pub(crate) Hadamard rotation state"),
];

fn is_undrivable(q: KvQuant) -> bool {
    let name = q.to_string();
    UNDRIVABLE.iter().any(|(n, _)| *n == name)
}

/// Every codec whose decode reads only the bf16 mirror — derived, not listed.
fn mirror_only_quants() -> Vec<KvQuant> {
    rmlx_kv_quant::ALL_KV_QUANTS
        .iter()
        .copied()
        .filter(|q| !q.materialises_packed_store() && *q != KvQuant::None && !is_undrivable(*q))
        .collect()
}

/// Every codec that keeps a packed store and can be driven through the public
/// prefill API.
fn store_keeping_quants() -> Vec<KvQuant> {
    rmlx_kv_quant::ALL_KV_QUANTS
        .iter()
        .copied()
        .filter(|q| q.materialises_packed_store() && !is_undrivable(*q))
        .collect()
}

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

/// Write one `.kvb` for `quant` into `dir` and return its **payload** size in
/// bytes — the file minus the safetensors header.
///
/// The distinction is load-bearing, not pedantry. The header is a JSON blob
/// carrying `META_KV_QUANT = kv_quant.to_string()`, so `"planar"` writes two
/// more bytes of metadata than `"k8v8"`. Comparing whole files makes a codec
/// *rename* — or one more metadata key — flip an equality that is supposed to
/// be about tensors, and the failure would read as "the packed store came
/// back" rather than "the header grew". Today they happen to agree only
/// because safetensors pads the header to an 8-byte multiple and both land in
/// the same bucket; that is luck, not a property.
///
/// Format: the first 8 bytes are a little-endian u64 header length; the payload
/// is everything after `8 + header_len`.
fn spill_payload_size(dir: &TempDir, label: &str, quant: KvQuant) -> u64 {
    let device = Device::Cpu;
    let cache = build_kvcache(quant);
    let hash = synthetic_hash(label);
    let path = dir.path().join(format!("{hash}.kvb"));
    rmlx_kv_ssd::write_caches(&path, device, MODEL_ID, quant, &[cache], &[]).unwrap();
    payload_bytes_of(&path, label)
}

/// Payload size of a safetensors file: total minus `8 + header_len`.
fn payload_bytes_of(path: &std::path::Path, label: &str) -> u64 {
    let total = std::fs::metadata(path)
        .unwrap_or_else(|e| panic!("metadata for {label}: {e}"))
        .len();
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {label}: {e}"));
    assert!(
        bytes.len() >= 8,
        "{label}: file is shorter than the safetensors header-length prefix"
    );
    let mut len_le = [0u8; 8];
    len_le.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(len_le);
    total
        .checked_sub(8 + header_len)
        .unwrap_or_else(|| panic!("{label}: header claims {header_len} B of a {total} B file"))
}

/// bf16 payload of one `[B, kv_h, S, D]` K/V pair: two tensors, 2 bytes per
/// element. Everything above this in the file is the safetensors JSON header.
fn bf16_payload_bytes() -> u64 {
    let n: u64 = (BATCH * KV_HEADS * BLOCK_TOKENS * HEAD_DIM) as u64;
    2 * n * 2
}

/// Every mirror-only quant spills the **same** bf16 block, byte-identical in
/// payload to the one a plain `KvQuant::None` cache spills.
///
/// This is the on-disk face of the `exit_prefill` store skip: those codecs hold
/// nothing but the bf16 mirror, so that mirror is the whole layer and it is
/// what gets persisted. Hydrate then hands the cache back the same bytes it
/// spilled, which is why a prompt-cache hit served from disk decodes
/// identically to one served from RAM.
///
/// The set is derived from the predicate, so a codec that joins the family
/// later is covered without anyone remembering to add it.
#[test]
fn mirror_only_quants_spill_the_same_bf16_block_as_none() {
    let tmp = TempDir::new().unwrap();
    let baseline = spill_payload_size(&tmp, "None", KvQuant::None);
    let payload = bf16_payload_bytes();

    assert_eq!(
        baseline, payload,
        "the bf16 block's payload must be exactly the two bf16 tensors"
    );

    let quants = mirror_only_quants();
    assert!(
        quants.len() >= 15,
        "the mirror-only family should be most of the codec surface; got {}",
        quants.len()
    );

    println!("\n── mirror_only_quants_spill_the_same_bf16_block_as_none ──");
    println!("  {:<30} {:>10} bytes", "None (oracle)", baseline);
    for quant in quants {
        let label = quant.to_string();
        let size = spill_payload_size(&tmp, &label, quant);
        println!("  {label:<30} {size:>10} bytes");
        assert_eq!(
            size, baseline,
            "{label} builds no packed store, so its block must carry the same \
             bf16 payload `None` spills ({baseline} B); got {size} B"
        );
    }
}

/// The controls: every codec that keeps a packed store spills that store — a
/// block that is neither the bf16 block nor an empty geometry stub.
///
/// Both comparisons are load-bearing. Against the bf16 block it proves the
/// mirror-only rule did not swallow a codec that needs its codes; against the
/// same codec's unfilled block it proves the codes actually reached the file.
/// Drop the second and a writer that emitted geometry for every layer would
/// satisfy the first by landing on the geometry-only size.
#[test]
fn store_keeping_quants_spill_their_codes_not_the_bf16_block() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let bf16_block = spill_payload_size(&tmp, "None", KvQuant::None);
    let n: u64 = (BATCH * KV_HEADS * BLOCK_TOKENS * HEAD_DIM) as u64;

    let quants = store_keeping_quants();
    assert!(
        !quants.is_empty(),
        "at least one drivable store-keeping codec must exist, or this control \
         proves nothing"
    );

    println!("\n── store_keeping_quants_spill_their_codes_not_the_bf16_block ──");
    println!("  {:<30} {:>10} bytes", "None (bf16 block)", bf16_block);
    for quant in quants {
        let label = quant.to_string();

        // Geometry-only reference: same codec, never filled, so there is no
        // payload to write.
        let empty_path = tmp.path().join(format!("empty_{label}.kvb"));
        rmlx_kv_ssd::write_caches(
            &empty_path,
            device,
            MODEL_ID,
            quant,
            &[KvCache::with_quant_max_seq(quant, 4096)],
            &[],
        )
        .unwrap();
        let geometry_only = payload_bytes_of(&empty_path, &label);

        let size = spill_payload_size(&tmp, &label, quant);
        println!("  {label:<30} {size:>10} bytes  (empty {geometry_only} B)");
        assert_ne!(
            size, bf16_block,
            "{label} keeps a packed store, so its block must not be the bf16 block"
        );
        // The K codes alone are at least half a byte per element; anything near
        // the geometry-only size means the payload never reached the file.
        assert!(
            size > geometry_only + n / 2,
            "{label} must spill its codes: got {size} B against a {geometry_only} B \
             geometry-only block for the same codec"
        );
    }
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

    let cases: Vec<KvQuant> = mirror_only_quants()
        .into_iter()
        .chain(store_keeping_quants())
        .collect();

    for quant in cases {
        let label = quant.to_string();
        let label = label.as_str();
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

/// Coverage check driven off the enum, not off a list this file maintains.
///
/// The previous form compared a hand-written `ALL_KNOWN` const against
/// `TESTED ∪ EXCLUDED`. Adding a `KvQuant` variant does not touch `ALL_KNOWN`,
/// so it could not fail for the one case it existed to catch — and it had
/// already drifted 21 variants behind the enum while still printing a
/// "N/N variants covered" line. Driving it off `ALL_KV_QUANTS`, which lives
/// beside the enum and is pinned exhaustive by `variant_index`, means a new
/// codec lands here as an unclassified name on the next run.
#[test]
fn every_kv_quant_is_tested_or_explicitly_excluded() {
    let mut covered: HashSet<String> = mirror_only_quants()
        .into_iter()
        .chain(store_keeping_quants())
        .map(|q| q.to_string())
        .collect();
    covered.insert(KvQuant::None.to_string());

    let unclassified: Vec<String> = rmlx_kv_quant::ALL_KV_QUANTS
        .iter()
        .map(ToString::to_string)
        .filter(|name| !covered.contains(name) && !UNDRIVABLE.iter().any(|(n, _)| n == name))
        .collect();

    // The undrivable list must also stay honest: a name that no longer exists
    // in the enum is a stale exemption quietly widening the hole.
    let live: HashSet<String> = rmlx_kv_quant::ALL_KV_QUANTS
        .iter()
        .map(ToString::to_string)
        .collect();
    for (name, _) in UNDRIVABLE {
        assert!(
            live.contains(*name),
            "UNDRIVABLE names '{name}', which is not a KvQuant any more — remove it"
        );
    }

    assert!(
        unclassified.is_empty(),
        "these KvQuant variants are neither driven nor excluded here: {unclassified:?}"
    );
    println!(
        "\nevery_kv_quant_is_tested_or_explicitly_excluded: {} variants, {} driven, {} undrivable",
        rmlx_kv_quant::ALL_KV_QUANTS.len(),
        covered.len(),
        UNDRIVABLE.len()
    );
}
