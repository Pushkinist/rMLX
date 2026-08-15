use super::*;
use crate::block_io::write_caches;
use rmlx_mlx::{Array, Device, Dtype};
use tempfile::TempDir;

const MODEL_ID: &str = "Qwen3ForCausalLM/test-snap";
const QUANT: KvQuant = KvQuant::K8V8;
/// a `layout_key` of zero drops the layout component of the seed, leaving the
/// model + codec partition. Fixtures here key their index rows through
/// [`cache_seed`], exactly as the RAM push side that produced the spilled
/// digests does; the layout salt is exercised separately by
/// `salted_keyed_block_is_found_by_probe`.
const TEST_LAYOUT_KEY: u64 = 0;
/// Non-zero on purpose: a zero signature would let a probe that forgot the
/// model term still match these fixtures, and the whole file would keep
/// passing against a hydrator that cannot find anything a real model spilled.
const TEST_MODEL_SIG: u64 = 0x00c0_ffee_0bad_f00d;

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

    // One full block (256 tokens) of prompt ids → one chained digest, keyed
    // with the production-truth salted seed (matches the probe in `lookup`).
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes_seeded(
        &prompt_ids,
        cache_seed(TEST_LAYOUT_KEY, QUANT, TEST_MODEL_SIG),
    );
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
    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        TEST_LAYOUT_KEY,
        TEST_MODEL_SIG,
        device,
        dir,
        index,
    );
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

/// Production-truth probe match: the spill side keys every index row with the
/// digest seed [`cache_seed`] builds from `(layout_key, kv_quant, model_sig)`.
/// A block recorded under that key, with a non-zero layout key, must be found
/// by `lookup` — i.e. the probe seeds the digest stream identically to the
/// spill side. (The other hydrate tests use `layout_key == 0`, so they pin the
/// codec + model components but cannot catch a layout-key mismatch; this test
/// pins all three.)
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: single full block yields exactly one chained digest, asserted before index"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn salted_keyed_block_is_found_by_probe() {
    // Non-zero layout key + the real codec salt, exactly as the spill side does.
    const LK: u64 = 0x1234_5678_9abc_def0;

    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    // One full block of prompt ids.
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();

    // Production-truth seed: the shared `cache_seed`, all three terms live.
    let salted = chained_block_hashes_seeded(&prompt_ids, cache_seed(LK, QUANT, TEST_MODEL_SIG));
    assert_eq!(salted.len(), 1);
    let key = hash_to_hex_local(salted[0]);

    // Build + spill a single-layer cache to <dir>/<key>.kvb, record the row
    // under the SALTED key + the non-zero layout key.
    let cache = build_kvcache(BLOCK_TOKENS as i32, 0x5A17);
    let path = dir.join(format!("{key}.kvb"));
    write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    index
        .record(&key, LK, &path, MODEL_ID, &QUANT.to_string(), size)
        .unwrap();

    // Hydrate with the same non-zero layout key → probe must produce the salted
    // digest and match the recorded row.
    let hydrator = SsdHydrator::with_index(MODEL_ID, QUANT, LK, TEST_MODEL_SIG, device, dir, index);
    let block = hydrator
        .lookup(&prompt_ids)
        .unwrap()
        .expect("salted-keyed SSD block must be found by the probe");
    assert_eq!(block.prompt_ids, prompt_ids, "matched prefix ids");
    assert_eq!(block.kv_caches.len(), 1, "one reconstructed layer");
}

/// The probe seed is the RAM push seed, and it partitions by model.
///
/// Both halves are asserted against the *same* on-disk row, because either one
/// alone is satisfiable by a broken hydrator:
///
/// 1. **Own model hits.** The row is keyed exactly as the RAM push side keys a
///    slot — `cache_seed(layout_key, kv_quant, model_sig)` with all three terms
///    non-trivial — and the hydrator carrying that `model_sig` must find it. A
///    probe that omits the model term computes a different digest stream, finds
///    nothing, and the tier 0-hits in silence: no error, just a full re-prefill
///    on every repeat.
/// 2. **Another model misses.** A hydrator for a *different* model over the
///    same namespace must not find that row. `--project` puts several models in
///    one `.kvb` directory, so the directory is not a per-model partition and
///    the seed is the only thing keeping them apart.
///
/// Half 1 alone passes for a probe that matches everything; half 2 alone passes
/// for a probe that matches nothing (which is precisely the defect). Together
/// they pin the seed.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: single full block yields exactly one chained digest, asserted before index"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn probe_finds_own_models_block_and_not_another_models() {
    /// Non-zero so the layout term is live alongside the model term.
    const LK: u64 = 0x0fed_cba9_8765_4321;
    /// A second model of the same arch, sharing the namespace.
    const OTHER_MODEL_SIG: u64 = 0x0123_4567_89ab_cdef;

    assert_ne!(
        cache_seed(LK, QUANT, TEST_MODEL_SIG),
        cache_seed(LK, QUANT, OTHER_MODEL_SIG),
        "the two fixtures must differ only in a term the seed actually uses"
    );

    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    // One full block, keyed the way the RAM push side keys the slot the spiller
    // later persists: through the shared seed, model term included.
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes_seeded(&prompt_ids, cache_seed(LK, QUANT, TEST_MODEL_SIG));
    assert_eq!(chained.len(), 1);
    let key = hash_to_hex_local(chained[0]);

    let cache = build_kvcache(BLOCK_TOKENS as i32, 0x0B1E);
    let path = dir.join(format!("{key}.kvb"));
    write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    index
        .record(&key, LK, &path, MODEL_ID, &QUANT.to_string(), size)
        .unwrap();

    // ── 1. The model that spilled it gets it back. ───────────────────────────
    let mine = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        LK,
        TEST_MODEL_SIG,
        device,
        dir.clone(),
        SsdKvIndex::open_at(&db).unwrap(),
    );
    let block = mine
        .lookup(&prompt_ids)
        .unwrap()
        .expect("a block keyed by the RAM push seed must be found by the probe");
    assert_eq!(block.prompt_ids, prompt_ids, "matched prefix ids");
    assert_eq!(block.kv_caches.len(), 1, "one reconstructed layer");

    // ── 2. A different model over the same namespace does not. ───────────────
    let theirs = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        LK,
        OTHER_MODEL_SIG,
        device,
        dir,
        SsdKvIndex::open_at(&db).unwrap(),
    );
    assert!(
        theirs.lookup(&prompt_ids).unwrap().is_none(),
        "another model's hydrator must not hydrate this model's K/V"
    );
}

/// `lookup_seeded` returns a `Vec<u64>` equal to the canonical seed recompute:
/// `chained_block_hashes_seeded(&block.prompt_ids, cache_seed(layout_key, QUANT, model_sig))`.
/// A non-zero layout key, the real codec salt and a non-zero model signature
/// exercise all three components of the seed.
///
/// The input is TWO full blocks but only the FIRST block is indexed/spilled, so
/// the matched prefix (`block.prompt_ids`) is one block while the input is two.
/// This pins that `lookup_seeded` recomputes over the MATCHED PREFIX, not the
/// full input — a regression hashing the input would return two digests and
/// fail `hashes.len() == 1`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: the first chained digest exists (full first block), and input_ids has 2*BLOCK_TOKENS elements so the [..BLOCK_TOKENS] slice is in range"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture setup: tempdir / index-open / write_caches / fs-metadata / record / lookup on a hermetic temp path; any failure is a test failure"
)]
fn lookup_seeded_matches_arch_recompute() {
    const LK: u64 = 0x1234_5678_9abc_def0;

    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    // TWO full blocks of input, but only the FIRST block is indexed below — so
    // the matched prefix is one block while the input is two.
    let input_ids: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    let first_block_ids = &input_ids[..BLOCK_TOKENS];

    // Key the index row with the FIRST block's canonical salted digest. The
    // chained hash is prefix-dependent, so the first digest of the 2-block input
    // equals the digest of the lone first block — the probe inside lookup finds
    // it after the (unindexed) 2-block digest misses.
    let salted =
        chained_block_hashes_seeded(first_block_ids, cache_seed(LK, QUANT, TEST_MODEL_SIG));
    assert_eq!(salted.len(), 1);
    let key = hash_to_hex_local(salted[0]);

    // Build + spill a single-block, single-layer cache, record the row.
    let cache = build_kvcache(BLOCK_TOKENS as i32, 0x9F3C);
    let path = dir.join(format!("{key}.kvb"));
    write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    index
        .record(&key, LK, &path, MODEL_ID, &QUANT.to_string(), size)
        .unwrap();

    let hydrator = SsdHydrator::with_index(MODEL_ID, QUANT, LK, TEST_MODEL_SIG, device, dir, index);
    let (block, hashes) = hydrator
        .lookup_seeded(&input_ids)
        .unwrap()
        .expect("lookup_seeded must find the first-block prefix of the 2-block input");

    // Only the first block matched.
    assert_eq!(
        block.prompt_ids.len(),
        BLOCK_TOKENS,
        "matched prefix is the single indexed block, not the full input"
    );
    assert_eq!(
        block.prompt_ids, first_block_ids,
        "matched prefix == first block"
    );
    // Recompute is over the MATCHED PREFIX (one block → one digest), not the
    // 2-block input (which would be two digests).
    assert_eq!(
        hashes.len(),
        1,
        "lookup_seeded recomputes over the matched prefix, not the full input"
    );
    let expected =
        chained_block_hashes_seeded(&block.prompt_ids, cache_seed(LK, QUANT, TEST_MODEL_SIG));
    assert_eq!(
        hashes, expected,
        "lookup_seeded block_hashes must equal the canonical seed recompute"
    );
    assert_eq!(block.kv_caches.len(), 1, "one reconstructed layer");
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

    // Key the row with the production-truth salted seed so the probe matches it.
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes_seeded(
        &prompt_ids,
        cache_seed(TEST_LAYOUT_KEY, QUANT, TEST_MODEL_SIG),
    );
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

    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        TEST_LAYOUT_KEY,
        TEST_MODEL_SIG,
        device,
        dir,
        index,
    );
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

    // The hydrator probes with K8V4, so key the row with K8V4's salt so the
    // probe finds it — the header (K8V8) mismatch then triggers deletion.
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let key = hash_to_hex_local(
        chained_block_hashes_seeded(
            &prompt_ids,
            cache_seed(TEST_LAYOUT_KEY, KvQuant::K8V4, TEST_MODEL_SIG),
        )[0],
    );
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
    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        KvQuant::K8V4,
        TEST_LAYOUT_KEY,
        TEST_MODEL_SIG,
        device,
        dir,
        index,
    );
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
        TEST_MODEL_SIG,
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
        TEST_MODEL_SIG,
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

    // Key the row with the production-truth salted seed so the probe matches it.
    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes_seeded(
        &prompt_ids,
        cache_seed(TEST_LAYOUT_KEY, QUANT, TEST_MODEL_SIG),
    );
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

    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        TEST_LAYOUT_KEY,
        TEST_MODEL_SIG,
        device,
        dir,
        index,
    );
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

// ── Budget enforcement racing in-flight hydrates ──────────────────────────

/// Enforcing the on-disk budget while requests are hydrating is the shape this
/// repo has been bitten by before: a cache write that quietly does the wrong
/// thing reports no error, and the run just produces different output. So the
/// property under test is not "it did not crash" — it is that **every block a
/// racing hydrate is handed is the block it asked for**, checked against the
/// K-dequant of the exact cache that was spilled under that prompt.
///
/// Why the invariant holds, and what would break it: a block is only reachable
/// through its own `(hash, layout_key)` row, `evict_lru_until` deletes that row
/// before the `.kvb` is unlinked, and a read of a vanished or half-written file
/// fails the header check and takes the existing corrupt-block path (drop the
/// row, `warn!`, miss). Every racing outcome therefore collapses to a miss and
/// a full prefill. A regression that reused a row's path after its own row was
/// gone, or that shared one file across two hashes, would surface here as a
/// content mismatch rather than as a crash.
///
/// Three threads model the production tier: the spill drain thread growing the
/// namespace, the budget pass shrinking it, and a request thread hydrating. The
/// pre-race pass pins that the content check can pass on real data, so a run in
/// which the race produced only misses cannot pass vacuously.
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
fn budget_enforcement_racing_hydrates_never_serves_a_foreign_block() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};

    const RACE_PASSES: usize = 40;
    const NS: &str = "race-ns";

    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let fx = Arc::new(seed_race_fixture(&dir, &db, device));
    // Sized so both arms of the race stay well populated: the writer's filler
    // keeps the namespace over the ceiling (so the budget pass always has work
    // and lookups keep missing), while the prompt blocks are not crowded down to
    // a survival rate that would let a slow box reach zero hits and skip the
    // content check entirely. A hit does touch its block to the back of the
    // eviction queue, but the writer records fresh filler continuously and every
    // one of those lands ahead of it, so a prompt block survives only until
    // enough newer filler arrives — the headroom is what keeps `hits > 0`
    // robust rather than lucky. Measured: ~32-42% hits over 240 lookups.
    let budget = fx.block_bytes * (RACE_PROMPTS as u64 * 4);

    // Pre-race pass on quiet state: every prompt hits and its content is its
    // own. This is what stops the racing assertions below from passing on a run
    // that only ever saw misses.
    {
        let hydrator = SsdHydrator::with_index(
            MODEL_ID,
            QUANT,
            RACE_LK,
            TEST_MODEL_SIG,
            device,
            dir.clone(),
            SsdKvIndex::open_at(&db).unwrap(),
        );
        for i in 0..RACE_PROMPTS {
            let block = hydrator
                .lookup(&fx.prompts[i])
                .unwrap()
                .expect("quiet-state lookup must hit");
            assert_own_block(&block, &fx, i, device, "quiet-state");
        }
    }

    let barrier = Arc::new(Barrier::new(3));
    let stop = Arc::new(AtomicU64::new(0));
    let evicted_total = Arc::new(AtomicU64::new(0));

    // Thread 1 — the budget pass, standing in for the spill drain thread's
    // post-write enforcement.
    let evictor = {
        let (db, barrier, stop, evicted_total) = (
            db.clone(),
            Arc::clone(&barrier),
            Arc::clone(&stop),
            Arc::clone(&evicted_total),
        );
        std::thread::spawn(move || {
            let idx = SsdKvIndex::open_at(&db).unwrap();
            barrier.wait();
            while stop.load(Ordering::Acquire) == 0 {
                let n = crate::ssd_tier::enforce_namespace_budget(&idx, NS, budget);
                evicted_total.fetch_add(n, Ordering::AcqRel);
                std::thread::yield_now();
            }
        })
    };

    // Thread 2 — namespace growth. Fresh keys every time, exactly as a spiller
    // records a block it has just written.
    let writer = {
        let (db, dir, barrier, stop, block_bytes) = (
            db.clone(),
            dir.clone(),
            Arc::clone(&barrier),
            Arc::clone(&stop),
            fx.block_bytes,
        );
        std::thread::spawn(move || {
            let idx = SsdKvIndex::open_at(&db).unwrap();
            barrier.wait();
            let mut n = 0u64;
            while stop.load(Ordering::Acquire) == 0 {
                let key = format!("filler{n:012x}");
                let p = dir.join(format!("{key}.kvb"));
                if std::fs::write(&p, vec![0u8; block_bytes as usize]).is_ok() {
                    let _ =
                        idx.record(&key, RACE_LK, &p, MODEL_ID, &QUANT.to_string(), block_bytes);
                }
                n += 1;
                std::thread::yield_now();
            }
        })
    };

    // Thread 3 — the request thread. Re-records its own block after a miss,
    // which is what a real miss leads to (prefill, then spill on eviction).
    let hydrator_thread = {
        let (barrier, fx) = (Arc::clone(&barrier), Arc::clone(&fx));
        std::thread::spawn(move || {
            let re_index = SsdKvIndex::open_at(&db).unwrap();
            let hydrator = SsdHydrator::with_index(
                MODEL_ID,
                QUANT,
                RACE_LK,
                TEST_MODEL_SIG,
                device,
                dir.clone(),
                SsdKvIndex::open_at(&db).unwrap(),
            );
            barrier.wait();
            let (mut hits, mut misses) = (0u64, 0u64);
            for _ in 0..RACE_PASSES {
                for i in 0..RACE_PROMPTS {
                    match hydrator.lookup(&fx.prompts[i]) {
                        Ok(Some(block)) => {
                            hits += 1;
                            assert_own_block(&block, &fx, i, device, "racing");
                        }
                        Ok(None) => {
                            misses += 1;
                            // Re-record from the bytes captured at seed time —
                            // no MLX work off the seeding thread.
                            let p = dir.join(format!("{}.kvb", fx.keys[i]));
                            if std::fs::write(&p, &fx.kvb_bytes[i]).is_ok() {
                                let _ = re_index.record(
                                    &fx.keys[i],
                                    RACE_LK,
                                    &p,
                                    MODEL_ID,
                                    &QUANT.to_string(),
                                    fx.kvb_bytes[i].len() as u64,
                                );
                            }
                        }
                        Err(e) => panic!("hydrate must never surface an error, got {e}"),
                    }
                }
            }
            (hits, misses)
        })
    };

    let (hits, misses) = hydrator_thread.join().expect("hydrator thread panicked");
    stop.store(1, Ordering::Release);
    evictor.join().expect("evictor thread panicked");
    writer.join().expect("writer thread panicked");

    // `assert_own_block` — the whole point of the test — only runs on a hit, so
    // a run that raced its way to zero hits would pass without ever checking a
    // block's content. Nothing else here catches that: `hits + misses` is
    // tautological, and `evicted_total` only shows the evictor ran.
    assert!(
        hits > 0,
        "no racing lookup hit ({hits} hits / {misses} misses), so the content \
         check never ran; the run proves nothing about which block a racing \
         hydrate is served"
    );
    assert_eq!(
        hits + misses,
        (RACE_PASSES * RACE_PROMPTS) as u64,
        "every racing lookup must resolve to a hit or a miss"
    );
    assert!(
        evicted_total.load(Ordering::Acquire) > 0,
        "the budget pass must have actually evicted during the race, \
         otherwise nothing was raced"
    );
}

/// Prompts (and therefore indexed blocks) in the racing-budget fixture.
const RACE_PROMPTS: usize = 6;
/// Layout key the racing-budget fixture spills and probes under.
const RACE_LK: u64 = 0x00c0_ffee_0000_0001;

/// One indexed block per prompt, plus everything a checker needs to tell those
/// blocks apart after a round trip through the tier.
struct RaceFixture {
    /// Token ids per prompt; one full block each.
    prompts: Vec<Vec<u32>>,
    /// Index/`.kvb` key per prompt (the salted chained digest).
    keys: Vec<String>,
    /// K dequant of the cache that was spilled under each prompt.
    expected_k: Vec<Vec<f32>>,
    /// Raw `.kvb` bytes per prompt, so a miss can be re-recorded without doing
    /// MLX work off the thread that built the caches.
    kvb_bytes: Vec<Vec<u8>>,
    /// On-disk size of one block, the unit the budget is expressed in.
    block_bytes: u64,
}

/// Seed `dir`'s index with one block per prompt. Each prompt gets its own token
/// ids *and* its own KV content, which is what makes "served the wrong block"
/// observable in the data rather than only in the token ids.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn seed_race_fixture(dir: &std::path::Path, db: &std::path::Path, device: Device) -> RaceFixture {
    let index = SsdKvIndex::open_at(db).unwrap();
    let prompts: Vec<Vec<u32>> = (0..RACE_PROMPTS)
        .map(|i| {
            let base = (i as u32 + 1) * 100_000;
            (base..base + BLOCK_TOKENS as u32).collect()
        })
        .collect();
    let mut keys = Vec::with_capacity(RACE_PROMPTS);
    let mut expected_k = Vec::with_capacity(RACE_PROMPTS);
    let mut kvb_bytes = Vec::with_capacity(RACE_PROMPTS);
    for (i, ids) in prompts.iter().enumerate() {
        let chained = chained_block_hashes_seeded(ids, cache_seed(RACE_LK, QUANT, TEST_MODEL_SIG));
        assert_eq!(chained.len(), 1);
        let key = hash_to_hex_local(chained[0]);
        let cache = build_kvcache(BLOCK_TOKENS as i32, 0x9E11 + i as u64 * 977);
        expected_k.push(probe_k(&cache, device));
        let path = dir.join(format!("{key}.kvb"));
        write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
        let size = std::fs::metadata(&path).unwrap().len();
        index
            .record(&key, RACE_LK, &path, MODEL_ID, &QUANT.to_string(), size)
            .unwrap();
        kvb_bytes.push(std::fs::read(&path).unwrap());
        keys.push(key);
    }
    let block_bytes = index.total_bytes().unwrap() / RACE_PROMPTS as u64;
    RaceFixture {
        prompts,
        keys,
        expected_k,
        kvb_bytes,
        block_bytes,
    }
}

/// Assert `block` is the block that was spilled under `fx.prompts[i]` — both
/// its token ids and its K content. Content is what catches a block served
/// under the wrong key; token ids alone would not, since they are recomputed
/// from the caller's own prompt.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn assert_own_block(
    block: &HydratedBlock,
    fx: &RaceFixture,
    i: usize,
    device: Device,
    whence: &str,
) {
    assert_eq!(
        &block.prompt_ids, &fx.prompts[i],
        "{whence}: hydrate served prompt {i} foreign token ids"
    );
    let got = probe_k(&block.kv_caches[0], device);
    assert_eq!(got.len(), fx.expected_k[i].len());
    let err = fx.expected_k[i]
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1e-3,
        "{whence}: hydrate served prompt {i} a block whose K content is not its own \
         (max err {err})"
    );
}

/// A row whose `.kvb` is gone — what a budget pass interrupted between its row
/// delete and its file unlink leaves behind, and what an operator's `rm` leaves
/// behind — must read as a clean miss that also repairs the row, and the
/// namespace must stay usable afterwards. Half-applied maintenance leaves a
/// working tier, not a poisoned one.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: single full block yields exactly one chained digest, asserted before index"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn hydrate_of_a_row_whose_file_vanished_is_a_miss_and_leaves_the_tier_usable() {
    const LK: u64 = 0x00c0_ffee_0000_0002;

    let device = Device::Cpu;
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let db = dir.join("index.db");
    let index = SsdKvIndex::open_at(&db).unwrap();

    let prompt_ids: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let chained = chained_block_hashes_seeded(&prompt_ids, cache_seed(LK, QUANT, TEST_MODEL_SIG));
    assert_eq!(chained.len(), 1);
    let key = hash_to_hex_local(chained[0]);
    let path = dir.join(format!("{key}.kvb"));

    let cache = build_kvcache(BLOCK_TOKENS as i32, 0x7E57);
    let expected = probe_k(&cache, device);
    write_caches(&path, device, MODEL_ID, QUANT, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    let kvb = std::fs::read(&path).unwrap();
    index
        .record(&key, LK, &path, MODEL_ID, &QUANT.to_string(), size)
        .unwrap();

    // Row survives, file does not — the half-applied state.
    std::fs::remove_file(&path).unwrap();

    let hydrator = SsdHydrator::with_index(
        MODEL_ID,
        QUANT,
        LK,
        TEST_MODEL_SIG,
        device,
        dir,
        SsdKvIndex::open_at(&db).unwrap(),
    );
    assert!(
        hydrator.lookup(&prompt_ids).unwrap().is_none(),
        "a row pointing at a vanished file must read as a miss"
    );
    assert!(
        index.lookup(&key, LK).unwrap().is_none(),
        "the dangling row must be dropped, not left to miss forever"
    );

    // The namespace is still usable: re-spill the same block and hydrate it.
    std::fs::write(&path, &kvb).unwrap();
    index
        .record(&key, LK, &path, MODEL_ID, &QUANT.to_string(), size)
        .unwrap();
    let block = hydrator
        .lookup(&prompt_ids)
        .unwrap()
        .expect("re-spilled block must hydrate");
    let got = probe_k(&block.kv_caches[0], device);
    let err = expected
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(err < 1e-3, "re-spilled block round-trip error {err}");
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
