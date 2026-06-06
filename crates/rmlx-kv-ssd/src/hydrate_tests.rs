use super::*;
use crate::block_io::write_caches;
use crate::hashing::chained_block_hashes;
use rmlx_mlx::{Array, Device, Dtype};
use tempfile::TempDir;

const MODEL_ID: &str = "Qwen3ForCausalLM/test-snap";
const QUANT: KvQuant = KvQuant::K8V8;
/// a `layout_key` of zero collapses the seeded digest stream to the
/// un-salted `chained_block_hashes` output (`FNV_OFFSET ^ 0 == FNV_OFFSET`),
/// so the pre-hydrate fixtures still work byte-for-byte.
const TEST_LAYOUT_KEY: u64 = 0;

// Deterministic LCG f32 data in [-1, 1] (same generator as block_io tests).
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

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: data is a &[f32] with known length; byte reinterpret valid for f32
    // (alignment ≥ 1, size = 4); total byte count = data.len() * 4 fits in isize.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

/// Build a single-layer K8V8 `KvCache` populated with `seq` tokens, by
/// driving the public `update` path (CPU). Returns the cache + the flat f32
/// K dequant for tolerance comparison.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn build_kvcache(seq: i32, seed: u64) -> KvCache {
    let device = Device::Cpu;
    let mut c = KvCache::with_quant_max_seq(QUANT, 4096);
    // shape [B=1, kv_h=2, seq, D=128]
    let shape = [1i32, 2, seq, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = arr(&lcg(n, seed), &shape);
    let v = arr(&lcg(n, seed ^ 0xABCD), &shape);
    c.enter_prefill();
    c.update(&k, &v, device).unwrap();
    c.exit_prefill(device).unwrap();
    c
}

/// (a): a block written by `write_caches` + recorded in the index is read
/// back by `SsdHydrator::lookup`, reconstructing a KV cache whose K dequant
/// matches the spilled one within the fp tolerance.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn ssd_hit_reconstructs_block_within_tolerance() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    // One full block (256 tokens) of prompt ids → one chained digest.
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes(&prompt_ids);
    assert_eq!(chained.len(), 1);
    let key = hash_to_hex_local(chained[0]);

    // Build + spill a single-layer cache to <dir>/<key>.kvb, record the row.
    let cache = build_kvcache(BLOCK_TOKENS as i32, 0x51D1);
    let before = probe_k(&cache, device);
    let path = dir.join(format!("{key}.kvb"));
    write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    index
        .record(
            &key,
            TEST_LAYOUT_KEY,
            &path,
            MODEL_ID,
            &QUANT.to_string(),
            size,
        )
        .unwrap();

    // Hydrate.
    let hydrator = SsdHydrator::with_index(MODEL_ID, QUANT, TEST_LAYOUT_KEY, device, dir, index);
    let block = hydrator
        .lookup(&prompt_ids)
        .unwrap()
        .expect("SSD hit expected");
    assert_eq!(block.prompt_ids, prompt_ids, "matched prefix ids");
    assert_eq!(block.kv_caches.len(), 1, "one reconstructed layer");
    assert_eq!(
        block.kv_caches[0].seq_len(),
        BLOCK_TOKENS as i32,
        "offset restored from header seq_len"
    );

    let after = probe_k(&block.kv_caches[0], device);
    assert_eq!(before.len(), after.len());
    let max_err = before
        .iter()
        .zip(&after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_err < 1e-3, "K dequant round-trip error {max_err}");
}

/// (b): a corrupt `.kvb` (garbled bytes) → the file + index row are deleted
/// and `lookup` returns `None` (fall back to prefill). No panic.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn corrupt_block_deletes_file_and_row_returns_miss() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes(&prompt_ids);
    let key = hash_to_hex_local(chained[0]);
    let path = dir.join(format!("{key}.kvb"));
    // Garbage that is not a valid safetensors file.
    std::fs::write(&path, b"not-a-safetensors-file").unwrap();
    index
        .record(
            &key,
            TEST_LAYOUT_KEY,
            &path,
            MODEL_ID,
            &QUANT.to_string(),
            22,
        )
        .unwrap();

    let hydrator = SsdHydrator::with_index(MODEL_ID, QUANT, TEST_LAYOUT_KEY, device, dir, index);
    let res = hydrator.lookup(&prompt_ids).unwrap();
    assert!(res.is_none(), "corrupt block must surface as a miss");
    assert!(!path.exists(), "corrupt .kvb file must be deleted");

    // Row deleted: a fresh index at the same db has no row.
    let idx2 = SsdKvIndex::open_at(&db).unwrap();
    assert!(
        idx2.lookup(&key, TEST_LAYOUT_KEY).unwrap().is_none(),
        "corrupt block's index row must be deleted"
    );
}

/// Metadata-mismatch (wrong kv_quant) is also treated as corruption: file +
/// row deleted, returns miss.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn metadata_mismatch_treated_as_corrupt() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let key = hash_to_hex_local(chained_block_hashes(&prompt_ids)[0]);
    // Write a valid block at K8V8 ...
    let cache = build_kvcache(BLOCK_TOKENS as i32, 0xBEEF);
    let path = dir.join(format!("{key}.kvb"));
    write_caches(&path, device, MODEL_ID, KvQuant::K8V8, &[cache], &[]).unwrap();
    index
        .record(&key, TEST_LAYOUT_KEY, &path, MODEL_ID, "k8v8", 0)
        .unwrap();

    // ... but hydrate expecting K8V4 → KvQuantMismatch inside read_caches.
    // unit test #8: layout_key matches but the .kvb header advertises
    // a different kv_quant; lookup must surface as miss AND delete file +
    // index row.
    let hydrator =
        SsdHydrator::with_index(MODEL_ID, KvQuant::K8V4, TEST_LAYOUT_KEY, device, dir, index);
    assert!(hydrator.lookup(&prompt_ids).unwrap().is_none());
    assert!(!path.exists(), "mismatched .kvb file must be deleted");

    // Row must also be gone (verify against a freshly reopened index DB).
    let idx2 = SsdKvIndex::open_at(&db).unwrap();
    assert!(
        idx2.lookup(&key, TEST_LAYOUT_KEY).unwrap().is_none(),
        "header-mismatch row must be deleted from the index"
    );
}

/// A miss (no indexed prefix) returns `None` without touching disk.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn no_indexed_prefix_is_miss() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let index = SsdKvIndex::open_at(&tmp.path().join("index.db")).unwrap();
    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        TEST_LAYOUT_KEY,
        device,
        tmp.path().to_path_buf(),
        index,
    );
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    assert!(hydrator.lookup(&prompt_ids).unwrap().is_none());
}

/// Sub-one-block prompts have no chained digest → never queried.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn short_prompt_never_queried() {
    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let index = SsdKvIndex::open_at(&tmp.path().join("index.db")).unwrap();
    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        TEST_LAYOUT_KEY,
        device,
        tmp.path().to_path_buf(),
        index,
    );
    let short: Vec<u32> = (0..(BLOCK_TOKENS as u32 - 1)).collect();
    assert!(hydrator.lookup(&short).unwrap().is_none());
}

/// SSD-tier observability (step2-D): after a successful `lookup` (SSD hit),
/// the event recorder sees exactly one `ssd_hydrate` row with `dur_us > 0`,
/// `bytes > 0`, and the five sub-phase fields present in the notes JSON.
///
/// Also verifies the existing `ssd_hit_reconstructs_block_within_tolerance`
/// semantics are preserved (block count and K-dequant round-trip).
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn ssd_hit_lookup_emits_hydrate_event() {
    use rmlx_metrics::events::EventRecorder;
    use rusqlite::Connection;

    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();
    let events_db_path = dir.join("events.db");

    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes(&prompt_ids);
    let key = hash_to_hex_local(chained[0]);
    let path = dir.join(format!("{key}.kvb"));

    // Build + write a valid K8V8 block.
    let cache = build_kvcache(BLOCK_TOKENS as i32, 0xAB12);
    write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    index
        .record(
            &key,
            TEST_LAYOUT_KEY,
            &path,
            MODEL_ID,
            &QUANT.to_string(),
            size,
        )
        .unwrap();

    // Open a hermetic event recorder.
    let rec =
        EventRecorder::open_at(&events_db_path, "hydrate-test-run").expect("open event recorder");

    let hydrator = SsdHydrator::with_index(MODEL_ID, QUANT, TEST_LAYOUT_KEY, device, dir, index);
    let block = hydrator
        .lookup_with_recorder(&prompt_ids, &rec)
        .unwrap()
        .expect("SSD hit expected");

    assert_eq!(block.kv_caches.len(), 1, "one reconstructed layer");
    assert_eq!(
        block.kv_caches[0].seq_len(),
        BLOCK_TOKENS as i32,
        "offset must be restored"
    );

    // Verify exactly one `ssd_hydrate` event row.
    let conn = Connection::open(&events_db_path).expect("reopen events DB");
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE op = 'ssd_hydrate'",
            [],
            |r| r.get(0),
        )
        .expect("query ssd_hydrate count");
    assert_eq!(count, 1, "exactly one ssd_hydrate event row expected");

    let (value, notes): (f64, String) = conn
        .query_row(
            "SELECT value, notes FROM events WHERE op = 'ssd_hydrate' LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("query ssd_hydrate row");

    assert!(value > 0.0, "dur_us must be > 0, got {value}");

    let parsed: serde_json::Value = serde_json::from_str(&notes).expect("notes must be valid JSON");
    let bytes = parsed["bytes"].as_u64().expect("notes.bytes must be u64");
    assert!(bytes > 0, "hydrated bytes must be > 0, got {bytes}");

    for field in [
        "dur_lookup_us",
        "dur_read_us",
        "dur_dequant_us",
        "dur_finalize_us",
        "dur_touch_us",
    ] {
        assert!(
            parsed.get(field).is_some(),
            "notes must have field '{field}'"
        );
    }

    let block_count = parsed["block_count"].as_u64().expect("notes.block_count");
    assert_eq!(block_count, 1, "one block reconstructed");
}

// ── helpers ───────────────────────────────────────────────────────────────

fn hash_to_hex_local(d: u64) -> String {
    format!("{d:016x}")
}

// Dequant the K side of a reconstructed/built K8V8 cache to flat f32.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: test only calls probe_k on K8V8 caches that always have a q8 K buffer; None return is a structural bug in the test"
)]
fn probe_k(c: &KvCache, device: Device) -> Vec<f32> {
    c.eval_gpu_state().unwrap();
    c.probe_k_dequant(device)
        .expect("probe_k: cache storage has no q8 K buffer (Paged/Mixed/None variants unsupported)")
}
