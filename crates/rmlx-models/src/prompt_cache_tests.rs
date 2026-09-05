use super::*;
use crate::kv_cache::kv_layer_quants;
use rmlx_core::error::Result;
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::chained_block_hashes;
use rmlx_mlx::{Array, Device};

// ── chained_block_hashes_seeded ────────────────────────────────────

/// unit test #2: `chained_block_hashes_seeded(ids, FNV_OFFSET)`
/// produces the exact same digests as the bare `chained_block_hashes(ids)`.
/// Guards the re-export contract — old call sites that have not opted into
/// layout-key salting must observe byte-identical behaviour.
/// [`request_cache_seed`] folds the mixture of the codec **the request is
/// running**, not a snapshot of the one the SSD tier attached under.
///
/// The layout key cannot carry this: it is computed once at attach from the
/// launch codec, while the per-request
/// `kv_quant` override both let a request run a different one. Pinned two ways
/// — the seed must equal the vector-form seed built from the request's own
/// codec, and it must move with the layer count that sizes that vector.
#[test]
fn request_cache_seed_folds_the_requests_own_mixture() {
    let lk = 0x0f0f_0f0f_0f0f_0f0f_u64;
    // Every codec a request can name for itself.
    for q in [KvQuant::K8V4, KvQuant::None, KvQuant::K8V8, KvQuant::Planar] {
        assert_eq!(
            request_cache_seed(lk, q, TEST_LAYERS, false, TEST_SIG),
            cache_seed(lk, q, &kv_layer_quants(TEST_LAYERS, q, false), TEST_SIG),
            "seed for {q} must be built from that codec's own per-layer mixture"
        );
    }
    assert_ne!(
        request_cache_seed(lk, KvQuant::K8V4, TEST_LAYERS, false, TEST_SIG),
        request_cache_seed(lk, KvQuant::K8V4, TEST_LAYERS + 4, false, TEST_SIG),
        "the layer count sizes the folded mixture, so it must reach the seed"
    );
}

#[test]
fn chained_seeded_with_fnv_offset_matches_bare() {
    let ids: Vec<u32> = (0..(3 * BLOCK_TOKENS as u32)).collect();
    let bare = chained_block_hashes(&ids);
    let seeded = chained_block_hashes_seeded(&ids, FNV_OFFSET);
    assert_eq!(bare.len(), 3, "3 full blocks → 3 chained digests");
    assert_eq!(seeded, bare, "seeded(FNV_OFFSET) must equal bare");
}

/// unit test #2 (extension): a non-`FNV_OFFSET` seed produces a
/// strictly different digest stream — the salt actually salts.
#[test]
fn chained_seeded_with_non_default_seed_diverges() {
    let ids: Vec<u32> = (0..(2 * BLOCK_TOKENS as u32)).collect();
    let bare = chained_block_hashes(&ids);
    let salted = chained_block_hashes_seeded(&ids, FNV_OFFSET ^ 0xdead_beef_cafe_babe);
    assert_eq!(bare.len(), salted.len(), "same block count");
    for (a, b) in bare.iter().zip(salted.iter()) {
        assert_ne!(a, b, "salted digest must diverge from bare digest");
    }
}

/// Minimal entry for testing: token ID list + chained block hashes,
/// no KV tensors. `truncated_to` records the last block-truncation call.
///
/// The consume-engine tests drive the three trait hooks through configurable
/// fields: `kv_quant` (so the engine's quant-guard can match a runtime quant),
/// `hydrate_complete` (gemma4-style incomplete-hydrate degrade), and
/// `reuse_kind` (the canned `is_reusable_prefix_of` result the engine consults
/// once its policy gate permits).
struct TestEntry {
    ids: Vec<u32>,
    hashes: Vec<u64>,
    truncated_to: std::cell::Cell<Option<usize>>,
    is_ssd_hydrated: bool,
    kv_quant: Option<KvQuant>,
    hydrate_complete: bool,
    reuse_kind: Option<ReuseKind>,
}

impl TestEntry {
    fn new(ids: Vec<u32>) -> Self {
        let hashes = chained_block_hashes(&ids);
        TestEntry {
            ids,
            hashes,
            truncated_to: std::cell::Cell::new(None),
            is_ssd_hydrated: false,
            kv_quant: None,
            hydrate_complete: true,
            reuse_kind: None,
        }
    }

    /// Issue #26: build an entry whose block-hash chain is salted with an
    /// explicit `seed` (the `FNV_OFFSET ^ layout_key ^ codec_salt` the
    /// production push uses). Lets the codec-partition test store a slot under
    /// one codec seed and query under another.
    fn new_seeded(ids: Vec<u32>, seed: u64) -> Self {
        let hashes = chained_block_hashes_seeded(&ids, seed);
        TestEntry {
            ids,
            hashes,
            truncated_to: std::cell::Cell::new(None),
            is_ssd_hydrated: false,
            kv_quant: None,
            hydrate_complete: true,
            reuse_kind: None,
        }
    }

    /// Mark this entry as reconstructed from the SSD tier (placeholder
    /// `first_id`). The Exact fast path must exclude it.
    fn ssd_hydrated(mut self) -> Self {
        self.is_ssd_hydrated = true;
        self
    }

    /// Build an entry whose block hashes are salted to match the engine's
    /// consume seed for `kv_quant` on a RAM-only run (`layout_key == 0`), and
    /// tag the entry with that same `kv_quant` so the quant-guard accepts it.
    fn for_quant(ids: Vec<u32>, kv_quant: KvQuant) -> Self {
        let seed = request_cache_seed(0, kv_quant, TEST_LAYERS, false, TEST_SIG);
        let hashes = chained_block_hashes_seeded(&ids, seed);
        TestEntry {
            ids,
            hashes,
            truncated_to: std::cell::Cell::new(None),
            is_ssd_hydrated: false,
            kv_quant: Some(kv_quant),
            hydrate_complete: true,
            reuse_kind: None,
        }
    }

    fn with_ssd_hydrated(mut self, v: bool) -> Self {
        self.is_ssd_hydrated = v;
        self
    }

    fn with_hydrate_complete(mut self, v: bool) -> Self {
        self.hydrate_complete = v;
        self
    }

    /// Set the canned `is_reusable_prefix_of` result. The engine only consults
    /// this once its policy gate permits (hydrated strict-prefix under any
    /// policy; non-hydrated partial under `Partial`).
    fn with_reuse_kind(mut self, kind: ReuseKind) -> Self {
        self.reuse_kind = Some(kind);
        self
    }
}

impl PromptCacheEntry for TestEntry {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.ids
    }

    fn block_hashes(&self) -> &[u64] {
        &self.hashes
    }

    fn deep_clone(&self) -> Result<Self> {
        Ok(TestEntry {
            ids: self.ids.clone(),
            hashes: self.hashes.clone(),
            truncated_to: std::cell::Cell::new(self.truncated_to.get()),
            is_ssd_hydrated: self.is_ssd_hydrated,
            kv_quant: self.kv_quant,
            hydrate_complete: self.hydrate_complete,
            reuse_kind: self.reuse_kind,
        })
    }

    // TestEntry holds no real caches — these satisfy the required accessors so
    // the trait's default bodies compile, but TestEntry keeps its own
    // `truncate_kv_to`/`kv_bytes` overrides below to exercise the OVERRIDE path
    // (recording the truncation length, faking byte counts). The default bodies
    // are covered by `RealEntry` further down.
    fn kv_caches(&self) -> &[KvCache] {
        &[]
    }

    fn kv_caches_mut(&mut self) -> &mut [KvCache] {
        &mut []
    }

    fn kv_quant(&self) -> Option<KvQuant> {
        self.kv_quant
    }

    fn is_ssd_hydrated(&self) -> bool {
        self.is_ssd_hydrated
    }

    fn lin_caches(&self) -> &[LinearAttnCache] {
        &[]
    }

    fn truncate_kv_to(&mut self, prefix_len: usize) -> Result<()> {
        self.truncated_to.set(Some(prefix_len));
        Ok(())
    }

    fn truncate_kv_to_block(&mut self, block_count: usize) -> Result<()> {
        self.truncated_to.set(Some(block_count * BLOCK_TOKENS));
        Ok(())
    }

    fn kv_bytes(&self) -> u64 {
        // Fake: 8 bytes per token id so we can assert non-zero bytes.
        self.ids.len() as u64 * 8
    }

    fn is_hydrate_complete(&self) -> bool {
        self.hydrate_complete
    }

    fn is_reusable_prefix_of(
        &self,
        prompt_ids: &[u32],
        _is_ssd_hydrated: bool,
        matched_blocks: usize,
    ) -> Option<ReuseKind> {
        // Mirror the real arch hooks' predicates so the matrix tests are honest;
        // the policy gate that decides WHEN to call this lives in the engine.
        match self.reuse_kind? {
            // moe HydratedTail / gemma4 B1: a STRICT token-level prefix only
            // (stored.len() < prompt.len() && starts_with). The strict-less is
            // the load-bearing guard for the hydrated-equal-length case — such
            // an entry returns `None` and falls to Miss even with a canned
            // reuse_kind, so its placeholder first_id is never replayed.
            kind @ ReuseKind::StrictPrefix { .. } => {
                if self.ids.len() < prompt_ids.len() && prompt_ids.starts_with(&self.ids) {
                    Some(kind)
                } else {
                    None
                }
            }
            // gemma4 block-truncate: a divergent partial that shares >= 1 leading
            // block (the engine passes find_best_prefix's block_count).
            kind @ ReuseKind::BlockTruncate { .. } => {
                if matched_blocks >= 1 {
                    Some(kind)
                } else {
                    None
                }
            }
        }
    }

    fn prepare_reuse(&self, kind: ReuseKind) -> Result<Self> {
        let cloned = self.deep_clone()?;
        if let ReuseKind::BlockTruncate { effective_blocks } = kind {
            cloned
                .truncated_to
                .set(Some(effective_blocks * BLOCK_TOKENS));
        }
        Ok(cloned)
    }
}

fn make_ids(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

fn entry(n: usize) -> TestEntry {
    TestEntry::new(make_ids(n))
}

// ── RealEntry — exercises the trait's DEFAULT method bodies ──────────────────
//
// `TestEntry` mocks `truncate_kv_to` / `kv_bytes`, so it cannot cover the
// default bodies introduced when those moved onto the trait. `RealEntry` holds
// genuine `KvCache` + `LinearAttnCache` instances and supplies ONLY the four
// required accessors, inheriting `truncate_kv_to` / `truncate_kv_to_block` /
// `kv_bytes` from the default impl.

const REAL_QUANT: KvQuant = KvQuant::K8V8;

/// Deterministic LCG f32 data in [-1, 1] (same generator as the kv-ssd hydrate
/// tests, so the byte shapes match the production spill path).
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
    reason = "test-only: Array::from_f32_slice cannot fail on a correctly-sized F32 buffer"
)]
fn arr(data: &[f32], shape: &[i32]) -> Array {
    Array::from_f32_slice(data, shape).unwrap()
}

/// Build a single-layer K8V8 `KvCache` populated with `seq` tokens via the
/// public prefill path on CPU (mirrors `kv-ssd::hydrate_tests::build_kvcache`).
#[allow(
    clippy::unwrap_used,
    reason = "test-only: CPU prefill of a correctly-shaped K8V8 cache is infallible here; any error is a structural bug in the fixture"
)]
fn build_kvcache(seq: i32, seed: u64) -> KvCache {
    let device = Device::Cpu;
    let mut c = KvCache::with_quant_max_seq(REAL_QUANT, 4096);
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

/// Entry built on real caches, supplying ONLY the required accessors so the
/// trait's default `truncate_kv_to` / `truncate_kv_to_block` / `kv_bytes`
/// bodies are exercised. No mock overrides.
struct RealEntry {
    ids: Vec<u32>,
    hashes: Vec<u64>,
    kv_caches: Vec<KvCache>,
    lin_caches: Vec<LinearAttnCache>,
    kv_quant: Option<KvQuant>,
}

impl RealEntry {
    /// `n_kv` real K8V8 caches of `seq` tokens each, plus `n_lin` empty linear
    /// caches. With `seq == 0` the caches have `offset() == 0` (never filled),
    /// which the default `truncate_kv_to` guard must skip without panic.
    fn new(ids: Vec<u32>, seq: i32, n_kv: usize, n_lin: usize) -> Self {
        let hashes = chained_block_hashes(&ids);
        let kv_caches = (0..n_kv)
            .map(|i| {
                if seq > 0 {
                    build_kvcache(seq, 0x51D1 ^ i as u64)
                } else {
                    // Never-prefilled cache → offset() == 0 → truncate guard skips.
                    KvCache::with_quant_max_seq(REAL_QUANT, 4096)
                }
            })
            .collect();
        let lin_caches = (0..n_lin).map(|_| LinearAttnCache::new()).collect();
        RealEntry {
            ids,
            hashes,
            kv_caches,
            lin_caches,
            kv_quant: Some(REAL_QUANT),
        }
    }
}

impl PromptCacheEntry for RealEntry {
    fn prompt_token_ids(&self) -> &[u32] {
        &self.ids
    }

    fn block_hashes(&self) -> &[u64] {
        &self.hashes
    }

    #[allow(
        clippy::unwrap_used,
        reason = "test-only: refcount deep-clone of CPU caches is infallible in this fixture"
    )]
    fn deep_clone(&self) -> Result<Self> {
        Ok(RealEntry {
            ids: self.ids.clone(),
            hashes: self.hashes.clone(),
            kv_caches: self
                .kv_caches
                .iter()
                .map(|c| c.try_deep_clone().unwrap())
                .collect(),
            lin_caches: self
                .lin_caches
                .iter()
                .map(|c| c.try_deep_clone().unwrap())
                .collect(),
            kv_quant: self.kv_quant,
        })
    }

    fn kv_caches(&self) -> &[KvCache] {
        &self.kv_caches
    }

    fn kv_caches_mut(&mut self) -> &mut [KvCache] {
        &mut self.kv_caches
    }

    fn kv_quant(&self) -> Option<KvQuant> {
        self.kv_quant
    }

    fn lin_caches(&self) -> &[LinearAttnCache] {
        &self.lin_caches
    }
    // truncate_kv_to / truncate_kv_to_block / kv_bytes: inherited from trait.
}

/// Default trait bodies over REAL `KvCache` instances:
///  - a 2-cache entry (offset > 0) reports `kv_bytes() > 0` and `truncate_kv_to`
///    actually shrinks the resident KV;
///  - a 2-cache entry whose caches were never prefilled (offset == 0) must NOT
///    panic on `truncate_kv_to` — the `offset() > 0` guard in the default body
///    skips them.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: fixture caches are constructed just above; deep_clone/eval are infallible on this CPU K8V8 fixture"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an unreachable rollback failure is the defect the test reports"
)]
fn default_truncate_and_kv_bytes_over_real_caches() {
    // 4 full blocks of tokens, caches each holding 4*BLOCK_TOKENS positions.
    let seq = (4 * BLOCK_TOKENS) as i32;
    let ids = make_ids(4 * BLOCK_TOKENS);
    let mut e = RealEntry::new(ids, seq, 2, 0);

    let before = e.kv_bytes();
    assert!(before > 0, "filled real caches must report non-zero bytes");

    // Truncate to 1 block (256 positions) via the DEFAULT body.
    e.truncate_kv_to(BLOCK_TOKENS)
        .expect("a full-attention entry rolls back to any prefix");
    for kv in e.kv_caches() {
        assert_eq!(
            kv.offset(),
            BLOCK_TOKENS as i32,
            "default truncate_kv_to must shrink each filled cache to the prefix length"
        );
    }
    let after = e.kv_bytes();
    assert!(
        after < before,
        "truncation must shrink resident KV ({after} < {before})"
    );

    // offset == 0 path: never-prefilled caches → default body's guard skips,
    // no panic, bytes are zero.
    let mut empty = RealEntry::new(make_ids(4 * BLOCK_TOKENS), 0, 2, 0);
    assert_eq!(empty.kv_bytes(), 0, "never-filled caches hold no bytes");
    empty
        .truncate_kv_to(BLOCK_TOKENS)
        .expect("the offset == 0 guard skips a never-filled cache");
    for kv in empty.kv_caches() {
        assert_eq!(kv.offset(), 0, "guard skipped the never-filled cache");
    }
}

/// Default `kv_bytes` sums BOTH KV `resident_bytes` and linear-attn
/// `resident_bytes`. Before this refactor only the qwen3.5-moe override summed
/// the lin half; the default body now makes that structural for every arch
/// that returns a non-empty `lin_caches()`.
#[test]
fn default_kv_bytes_sums_lin_caches() {
    let seq = (2 * BLOCK_TOKENS) as i32;
    let ids = make_ids(2 * BLOCK_TOKENS);
    // 1 real KvCache + 1 LinearAttnCache. Populate the lin state with real
    // arrays so `resident_bytes() > 0` — an empty lin cache reports 0, which would
    // make this test vacuous (it would pass even if the default body dropped the
    // `+ lin_caches()` term).
    let mut e = RealEntry::new(ids, seq, 1, 1);
    if let Some(lin) = e.lin_caches.first_mut() {
        // conv_state [B, kernel-1, conv_dim] bf16; delta_state [B, Hv, Dv, Dk] f32.
        lin.conv_state = Some(arr(&lcg(2 * 3 * 4, 0xC0), &[2, 3, 4]));
        lin.delta_state = Some(arr(&lcg(2 * 2 * 4 * 4, 0xD0), &[2, 2, 4, 4]));
    }

    let kv_only: u64 = e.kv_caches().iter().map(KvCache::resident_bytes).sum();
    let lin_only: u64 = e
        .lin_caches()
        .iter()
        .map(LinearAttnCache::resident_bytes)
        .sum();
    assert!(kv_only > 0, "the real KV cache contributes bytes");
    assert!(
        lin_only > 0,
        "the populated lin cache must contribute bytes"
    );
    assert_eq!(
        e.kv_bytes(),
        kv_only + lin_only,
        "default kv_bytes must equal KV resident_bytes + lin resident_bytes"
    );
}

/// 3 requests: 1st is a cold miss (cache empty), 2nd is a full-prompt
/// block-aligned hit (same prompt), 3rd is a partial prefix hit (longer
/// prompt sharing the leading blocks).
///
/// Expected counters after all 3 lookups:
/// hits = 2 (full hit + partial prefix hit)
/// misses = 1 (cold miss)
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn hit_miss_counters_three_requests() {
    // 2 full blocks (block-aligned so the whole prompt is matchable).
    let prompt_a: Vec<u32> = make_ids(2 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);

    // --- Request 1: cold miss (cache empty) ---
    let r1 = cache.find_best_prefix(&prompt_a, FNV_OFFSET);
    assert!(r1.is_none(), "r1 should be a miss");
    // Populate cache so subsequent requests can hit.
    cache.push(TestEntry::new(prompt_a.clone()));

    // --- Request 2: full-prompt block hit (same prompt) ---
    let r2 = cache.find_best_prefix(&prompt_a, FNV_OFFSET);
    assert!(r2.is_some(), "r2 should be a hit");
    let (_, blocks_2) = r2.unwrap();
    assert_eq!(blocks_2, 2, "full hit must cover all 2 blocks");

    // --- Request 3: partial prefix hit (one extra full block of new ids) ---
    let mut prompt_b = prompt_a.clone();
    prompt_b.extend(1000..1000 + BLOCK_TOKENS as u32);
    let r3 = cache.find_best_prefix(&prompt_b, FNV_OFFSET);
    assert!(r3.is_some(), "r3 should be a partial prefix hit");
    let (_, blocks_3) = r3.unwrap();
    assert_eq!(
        blocks_3, 2,
        "partial hit must match the 2 stored leading blocks only"
    );

    // --- Assert counters ---
    let stats = cache.stats();
    assert_eq!(
        stats.hits, 2,
        "expected 2 hits (r2 + r3), got {}",
        stats.hits
    );
    assert_eq!(
        stats.misses, 1,
        "expected 1 miss (r1), got {}",
        stats.misses
    );

    // --- Assert bytes > 0 (slot has 1 entry with prompt_a.len() * 8 bytes) ---
    assert!(
        stats.bytes > 0,
        "bytes should be non-zero after a slot is populated, got {}",
        stats.bytes
    );
    let expected_bytes = prompt_a.len() as u64 * 8;
    assert_eq!(
        stats.bytes, expected_bytes,
        "bytes should match kv_bytes() of the single slot"
    );
}

/// `is_ssd_hydrated()` defaults to `false` for a normal RAM-cached entry and
/// flips to `true` once an entry is marked hydrated.
#[test]
fn is_ssd_hydrated_defaults_false() {
    let ram = TestEntry::new(make_ids(BLOCK_TOKENS));
    assert!(
        !ram.is_ssd_hydrated(),
        "a normal RAM-cached entry must report is_ssd_hydrated() == false"
    );
    let hydrated = TestEntry::new(make_ids(BLOCK_TOKENS)).ssd_hydrated();
    assert!(
        hydrated.is_ssd_hydrated(),
        "an SSD-hydrated entry must report is_ssd_hydrated() == true"
    );
}

/// Pins the exact-hit selection predicate the per-arch generate loops evaluate:
/// an Exact serve requires `!entry.is_ssd_hydrated() && entry.prompt_token_ids()
/// == prompt`. A normal RAM entry whose ids equal the prompt is served (replay
/// the stored real first token); an SSD-hydrated entry with the SAME block-
/// aligned ids must NOT be served as Exact — its `first_id` is the placeholder 0
/// and replaying it poisons generation. Excluding it forces a fall-through to a
/// full re-prefill that recomputes the real first token.
///
/// This is the codec-agnostic core of the SSD exact-hit sentinel fix; it runs in
/// the default gate (no model) by checking the predicate against `find_best_prefix`
/// hits for both a RAM-cached and a hydrated full-prompt entry.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "find_best_prefix returns Some by construction (a block-aligned full-prompt entry was just pushed); the returned slot index is a valid `slots` index"
)]
fn ssd_hydrated_entry_excluded_from_exact_serve() {
    // Block-aligned full prompt (no tail) — the SSD exact-hit reproduction shape
    // where the restored prefix equals the whole prompt.
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);

    // The decision the generate loop makes on a matched slot.
    let served_as_exact = |cache: &mut PromptCache<TestEntry>, prompt: &[u32]| -> bool {
        let Some((slot_idx, _)) = cache.find_best_prefix(prompt, FNV_OFFSET) else {
            return false;
        };
        let entry = &cache.slots[slot_idx].entry;
        !entry.is_ssd_hydrated() && entry.prompt_token_ids() == prompt
    };

    // RAM-cached entry with the same ids → SERVED as Exact (real first token).
    let mut ram_cache: PromptCache<TestEntry> = PromptCache::new(4);
    ram_cache.push(TestEntry::new(prompt.clone()));
    assert!(
        served_as_exact(&mut ram_cache, &prompt),
        "a normal RAM-cached full-prompt entry must be served as Exact"
    );

    // SSD-hydrated entry with the same ids → NOT served as Exact (falls through
    // to re-prefill so the placeholder first_id is never replayed).
    let mut ssd_cache: PromptCache<TestEntry> = PromptCache::new(4);
    ssd_cache.push(TestEntry::new(prompt.clone()).ssd_hydrated());
    // Sanity: the block-hash match still fires (only the exact-serve gate differs).
    assert!(
        ssd_cache.find_best_prefix(&prompt, FNV_OFFSET).is_some(),
        "a hydrated full-prompt entry must still produce a block-hash match"
    );
    assert!(
        !served_as_exact(&mut ssd_cache, &prompt),
        "an SSD-hydrated full-prompt entry must NOT be served as Exact — \
         replaying its placeholder first_id poisons generation"
    );
}

/// Verify that a shared prefix of fewer than one full block is a miss.
#[test]
fn short_prefix_treated_as_miss() {
    let short_prompt: Vec<u32> = make_ids(BLOCK_TOKENS - 1);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(short_prompt.clone()));

    // Query with the same short prompt — no full block stored, so there is
    // nothing to match: must be a miss.
    let result = cache.find_best_prefix(&short_prompt, FNV_OFFSET);
    assert!(
        result.is_none(),
        "shared prefix < one 256-token block must be treated as a miss"
    );

    let stats = cache.stats();
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 1);
}

// --- M29 LRU tests ---

/// LRU ordering: push 3 entries (A, B, C) into a capacity-3 cache, then
/// hit B and C. Push D — A should be evicted (oldest unused).
#[test]
fn lru_bump_on_hit_determines_eviction_order() {
    let n = 2 * BLOCK_TOKENS; // 2 full blocks, block-aligned
                              // Build 4 distinct prompts that do NOT prefix-match each other.
                              // Interleave by 4 so even block 0 differs across prompts.
    let ids_a: Vec<u32> = (0..n as u32).map(|x| x * 4).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|x| x * 4 + 1).collect();
    let ids_c: Vec<u32> = (0..n as u32).map(|x| x * 4 + 2).collect();
    let ids_d: Vec<u32> = (0..n as u32).map(|x| x * 4 + 3).collect();

    let mut cache: PromptCache<TestEntry> =
        PromptCache::with_max_bytes(3, u64::MAX /* no RAM cap */);

    // Push A, B, C — fills to capacity.
    cache.push(TestEntry::new(ids_a.clone()));
    cache.push(TestEntry::new(ids_b.clone()));
    cache.push(TestEntry::new(ids_c.clone()));
    assert_eq!(cache.slots.len(), 3);

    // Hit B — B becomes MRU.
    let hit_b = cache.find_best_prefix(&ids_b, FNV_OFFSET);
    assert!(hit_b.is_some(), "B must hit");

    // Hit C — C becomes MRU.
    let hit_c = cache.find_best_prefix(&ids_c, FNV_OFFSET);
    assert!(hit_c.is_some(), "C must hit");

    // Push D — slot count cap triggers, A must be evicted (smallest seq).
    cache.push(TestEntry::new(ids_d.clone()));
    assert_eq!(cache.slots.len(), 3, "capacity still 3");

    let stats = cache.stats();
    assert_eq!(stats.evictions, 1, "exactly 1 eviction");

    // A must be gone; B, C, D must still be present.
    let still_has = |cache: &PromptCache<TestEntry>, ids: &[u32]| {
        cache
            .slots
            .iter()
            .any(|s| s.entry.prompt_token_ids() == ids)
    };
    assert!(!still_has(&cache, &ids_a), "A must have been evicted (LRU)");
    assert!(still_has(&cache, &ids_b), "B must still be present");
    assert!(still_has(&cache, &ids_c), "C must still be present");
    assert!(still_has(&cache, &ids_d), "D must be present");
}

/// RAM-cap eviction: set max_bytes below what two large entries would occupy.
/// First entry fits. Pushing the second should evict the first.
#[test]
fn ram_cap_eviction_triggers_correctly() {
    let n = 2 * BLOCK_TOKENS; // 512 tokens → 512*8 = 4096 bytes per entry
    let bytes_per_entry = n as u64 * 8;

    // Allow only 1.5× one entry — second push must evict the first.
    let max_bytes = bytes_per_entry + bytes_per_entry / 2;

    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(100, max_bytes);

    cache.push(entry(n));
    assert_eq!(cache.slots.len(), 1, "first entry fits");

    cache.push(entry(n));
    // Second entry (also bytes_per_entry) would push total to 2×bytes_per_entry > max_bytes.
    // So the first must be evicted.
    assert_eq!(cache.slots.len(), 1, "RAM cap: only one entry survives");

    let stats = cache.stats();
    assert_eq!(stats.evictions, 1, "one RAM-cap eviction");
}

/// Slot-count cap is still the hard ceiling even when RAM cap is large.
#[test]
fn slot_count_cap_still_hard_ceiling() {
    let n = 2 * BLOCK_TOKENS;
    let mut cache: PromptCache<TestEntry> =
        PromptCache::with_max_bytes(2, u64::MAX /* no RAM cap */);

    cache.push(entry(n));
    cache.push(entry(n));
    assert_eq!(cache.slots.len(), 2);

    cache.push(entry(n)); // triggers slot-count eviction
    assert_eq!(cache.slots.len(), 2, "slot cap held at 2");

    let stats = cache.stats();
    assert_eq!(stats.evictions, 1, "one slot-cap eviction");
}

/// `evictions` counter increments correctly across both eviction paths.
#[test]
fn evictions_counter_tracked() {
    let n = 2 * BLOCK_TOKENS;
    let bytes_per = n as u64 * 8;
    // RAM cap = exactly 2 entries; slot cap = 10 (won't interfere).
    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(10, bytes_per * 2);

    // Fill 2 entries — no eviction.
    cache.push(entry(n));
    cache.push(entry(n));
    assert_eq!(cache.stats().evictions, 0);

    // Push 3rd — RAM cap triggers, evicts 1.
    cache.push(entry(n));
    assert_eq!(cache.stats().evictions, 1, "one eviction after 3rd push");

    // Push 4th — RAM cap triggers again, evicts 1 more.
    cache.push(entry(n));
    assert_eq!(cache.stats().evictions, 2, "two evictions after 4th push");
}

/// Over-cap admission guard: a single snapshot whose KV alone exceeds the RAM
/// cap must NOT be admitted. Admitting it violates the cap and, on the next
/// warm (Exact) hit, forces a second full-size residency (deep_clone +
/// copy-on-write of the whole KV on the first decode append) — the warm-cache
/// decode stall on large-KV codecs. Refusing admission bounds peak memory to a
/// single live copy and makes the repeat request re-prefill, exactly like the
/// cold request (warm ≈ cold). Model- and codec-agnostic.
#[test]
fn over_cap_snapshot_is_not_admitted() {
    let n = 2 * BLOCK_TOKENS; // 512 ids → kv_bytes() = 512 * 8 = 4096
    let entry_bytes = n as u64 * 8;

    // Cap strictly below one entry → the entry alone cannot fit.
    let tiny_cap = entry_bytes - 1;
    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(4, tiny_cap);

    let prompt = make_ids(n);
    // Cold "store": the over-cap entry is refused admission.
    let stored = cache.push(TestEntry::new(prompt.clone()));
    assert!(
        stored.is_none(),
        "an over-cap snapshot must be refused admission"
    );
    assert_eq!(
        cache.slots.len(),
        0,
        "no slot is created for an over-cap snapshot"
    );

    // Warm request for the SAME prompt: a Miss (re-prefill), NOT an Exact hit on
    // an over-cap slot — the invariant that removes the warm-cache stall.
    let warm = cache.find_best_prefix(&prompt, FNV_OFFSET);
    assert!(
        warm.is_none(),
        "warm request must miss (re-prefill), not reuse an over-cap slot"
    );

    // An entry that DOES fit the cap is still admitted (guard is exactly at the
    // boundary: `new_bytes > max_bytes`, so `== max_bytes` fits).
    let mut ok_cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(4, entry_bytes);
    let ok_stored = ok_cache.push(TestEntry::new(prompt.clone()));
    assert_eq!(ok_stored, Some(0), "an entry that fits the cap is admitted");
    assert!(
        ok_cache.find_best_prefix(&prompt, FNV_OFFSET).is_some(),
        "a fitting entry is reusable"
    );
}

/// Refusing an over-cap snapshot must leave existing (valid, smaller) slots
/// intact — the guard never evicts to make room for something that still could
/// not fit.
#[test]
fn over_cap_refusal_preserves_existing_slots() {
    let small_n = BLOCK_TOKENS; // 256 ids → 2048 bytes
    let small_bytes = small_n as u64 * 8;
    // Cap holds the small entry but not the large one.
    let cap = small_bytes + 100;
    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(4, cap);

    let small_prompt: Vec<u32> = (0..small_n as u32).map(|x| x * 2).collect();
    assert_eq!(
        cache.push(TestEntry::new(small_prompt.clone())),
        Some(0),
        "the small entry fits and is admitted"
    );

    // Over-cap entry (distinct prompt so it cannot prefix-match): refused, and
    // the pre-existing small slot survives with no eviction.
    let big_prompt: Vec<u32> = (0..(4 * BLOCK_TOKENS) as u32).map(|x| x * 2 + 1).collect();
    assert!(
        cache.push(TestEntry::new(big_prompt)).is_none(),
        "the over-cap entry is refused admission"
    );
    assert_eq!(
        cache.slots.len(),
        1,
        "the valid small slot is left untouched by the over-cap refusal"
    );
    assert_eq!(
        cache.stats().evictions,
        0,
        "an over-cap refusal evicts nothing"
    );
    assert!(
        cache.find_best_prefix(&small_prompt, FNV_OFFSET).is_some(),
        "the small slot is still reusable after the refusal"
    );
}

/// C1: long-prompt block-aligned partial hit.
///
/// Cache a 4096-token prompt (16 full 256-tok blocks). Query with a
/// prompt that shares the first 3840 tokens (15 blocks) then has 256
/// different tokens. Expect a hit with `block_count == 15`, and the
/// cloned entry truncated to 3840 tokens.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an unreachable rollback failure is the defect the test reports"
)]
fn long_prompt_block_aligned_partial_hit() {
    let prompt_a: Vec<u32> = make_ids(16 * BLOCK_TOKENS); // 4096 tokens
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(prompt_a.clone()));

    // First 15 blocks (3840 tokens) identical, last block differs.
    let mut prompt_b = prompt_a[..15 * BLOCK_TOKENS].to_vec();
    prompt_b.extend(9000..9000 + BLOCK_TOKENS as u32);
    assert_eq!(prompt_b.len(), 16 * BLOCK_TOKENS);

    let r = cache.find_best_prefix(&prompt_b, FNV_OFFSET);
    assert!(r.is_some(), "must hit on the shared 15-block prefix");
    let (slot_idx, block_count) = r.unwrap();
    assert_eq!(block_count, 15, "must match exactly 15 leading blocks");

    // Block-aligned truncation lands at 3840 tokens.
    let mut cloned = cache.slots[slot_idx].entry.deep_clone().unwrap();
    cloned
        .truncate_kv_to_block(block_count)
        .expect("a full-attention entry rolls back to a block boundary");
    assert_eq!(
        cloned.truncated_to.get(),
        Some(15 * BLOCK_TOKENS),
        "entry must truncate to 3840 tokens (15 blocks)"
    );

    let stats = cache.stats();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 0);
}

/// C6: block-level counters — partial hit, full block hit, and total miss.
///
/// Cache a 16-block (4096-token) prompt.
///
/// (a) Partial-hit query: 15 shared blocks + 1 new block →
/// block_hits==15, block_misses==1, partial_hits==1, hits==1, misses==0.
///
/// (b) Full-block-match query: shares all 16 blocks (identical first 16
/// blocks) → partial_hits unchanged (not a partial hit), block_hits==16
/// added on top of previous.
///
/// (c) Total-miss query: no shared block → block_misses += want_blocks,
/// misses==1.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn block_level_counters_partial_and_full() {
    // 16 full blocks (4096 tokens).
    let base: Vec<u32> = make_ids(16 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(base.clone()));

    // (a) Partial hit: first 15 blocks match, 16th differs.
    let mut partial_prompt = base[..15 * BLOCK_TOKENS].to_vec();
    partial_prompt.extend(9000..9000 + BLOCK_TOKENS as u32);
    assert_eq!(partial_prompt.len(), 16 * BLOCK_TOKENS);

    let r_a = cache.find_best_prefix(&partial_prompt, FNV_OFFSET);
    assert!(r_a.is_some(), "(a) must be a hit");
    let (_, blocks_a) = r_a.unwrap();
    assert_eq!(blocks_a, 15, "(a) 15 blocks matched");

    let s = cache.stats();
    assert_eq!(s.hits, 1, "(a) hits");
    assert_eq!(s.misses, 0, "(a) misses");
    assert_eq!(s.block_hits, 15, "(a) block_hits");
    assert_eq!(s.block_misses, 1, "(a) block_misses");
    assert_eq!(s.partial_hits, 1, "(a) partial_hits");

    // (b) Full block match: exact same first 16 blocks.
    // Build a prompt that has the same 16 blocks then 256 extra new tokens.
    let mut full_prompt = base.clone();
    full_prompt.extend(20000..20000 + BLOCK_TOKENS as u32);
    assert_eq!(full_prompt.len(), 17 * BLOCK_TOKENS);

    // `find_best_prefix` only counts full 256-token blocks in `want`.
    // The want vector for this prompt has 17 hashes; cache has 16.
    // Matched = 16 (full match of stored entry), unmatched = 1.
    // Because best_blocks (16) < want_blocks (17) → still a partial_hit.
    let r_b = cache.find_best_prefix(&full_prompt, FNV_OFFSET);
    assert!(r_b.is_some(), "(b) must be a hit");
    let (_, blocks_b) = r_b.unwrap();
    assert_eq!(blocks_b, 16, "(b) 16 blocks matched");

    let s = cache.stats();
    assert_eq!(s.hits, 2, "(b) hits");
    assert_eq!(s.block_hits, 15 + 16, "(b) cumulative block_hits");
    // block_misses: (a) had 1, (b) has want_blocks-best_blocks = 17-16 = 1
    assert_eq!(s.block_misses, 1 + 1, "(b) cumulative block_misses");
    // partial_hits: (a)=1 + (b)=1 (16 < 17 → partial)
    assert_eq!(s.partial_hits, 2, "(b) partial_hits after second partial");

    // Build a truly full match (same 16-block prompt, no extension).
    // want_blocks == 16, best_blocks == 16 → NOT partial.
    let r_b2 = cache.find_best_prefix(&base, FNV_OFFSET);
    assert!(r_b2.is_some(), "(b2) must be a hit");
    let (_, blocks_b2) = r_b2.unwrap();
    assert_eq!(blocks_b2, 16, "(b2) 16 blocks matched");

    let s = cache.stats();
    assert_eq!(s.hits, 3, "(b2) hits");
    assert_eq!(s.block_hits, 15 + 16 + 16, "(b2) cumulative block_hits");
    assert_eq!(s.block_misses, 2, "(b2) block_misses unchanged (0 new)");
    assert_eq!(
        s.partial_hits, 2,
        "(b2) partial_hits unchanged (full match)"
    );

    // (c) Total miss: use an unrelated 4-block prompt with completely
    // different token ids (offset by 100000 to avoid any block collision).
    let miss_prompt: Vec<u32> = (100000..100000 + 4 * BLOCK_TOKENS as u32).collect();
    let r_c = cache.find_best_prefix(&miss_prompt, FNV_OFFSET);
    assert!(r_c.is_none(), "(c) must be a miss");

    let s = cache.stats();
    assert_eq!(s.misses, 1, "(c) misses");
    // want_blocks for miss_prompt = 4
    assert_eq!(
        s.block_misses,
        2 + 4,
        "(c) block_misses += 4 (want_blocks of miss)"
    );
    assert_eq!(s.partial_hits, 2, "(c) partial_hits unchanged after miss");
}

/// correctness: a hot-cache hit must not alter the prompt identity.
///
/// The reuse contract is: an Exact hit returns a `deep_clone` of the slot
/// whose `prompt_token_ids()` are byte-identical to the request prompt, so
/// the decode loop continues from a snapshot produced by *exactly* the same
/// tokens it would have prefilled cold. This unit-level invariant is what
/// guarantees a cache hit yields the same tokens as a cold run; the live
/// GPU hit-vs-cold byte-identity check is run separately by the supervisor.
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
fn exact_hit_clone_preserves_prompt_identity() {
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(prompt.clone()));

    let (idx, blocks) = cache
        .find_best_prefix(&prompt, FNV_OFFSET)
        .expect("identical prompt must hit");
    assert_eq!(blocks, 2, "full prompt covers both blocks");
    // Exact-hit gate at the call site: stored ids == request ids.
    assert_eq!(
        cache.slots[idx].entry.prompt_token_ids(),
        &prompt[..],
        "matched slot's tokens must equal the request prompt byte-for-byte"
    );
    let cloned = cache.slots[idx].entry.deep_clone().unwrap();
    assert_eq!(
        cloned.prompt_token_ids(),
        &prompt[..],
        "deep_clone must preserve prompt identity — reuse cannot change input"
    );
}

// --- spill-sink tests (generic, no disk, no model) ---

/// Mock spill sink: records the spilled entries' last block-hash into a
/// shared vec. Returns instantly — proves `push` does no I/O inline.
struct MockSink {
    captured: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
}

impl SpillSink<TestEntry> for MockSink {
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn spill(&self, entry: &TestEntry) {
        let key = entry.block_hashes().last().copied().unwrap_or(0);
        self.captured.lock().unwrap().push(key);
    }
}

/// (a): RAM-cap eviction with a sink attached captures the spilled
/// entry. The first entry is evicted on the second push and its block hash
/// must appear in the sink's capture log.
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn spill_sink_captures_ram_cap_eviction() {
    let n = 2 * BLOCK_TOKENS;
    let bytes_per_entry = n as u64 * 8;
    let max_bytes = bytes_per_entry + bytes_per_entry / 2; // ~1.5 entries

    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(100, max_bytes);
    cache.set_spill_sink(Box::new(MockSink {
        captured: captured.clone(),
    }));

    // Distinct prompts so their chained hashes differ.
    let ids_a: Vec<u32> = (0..n as u32).map(|x| x * 2).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|x| x * 2 + 1).collect();
    let expected_a = *chained_block_hashes(&ids_a).last().unwrap();

    cache.push(TestEntry::new(ids_a));
    assert!(captured.lock().unwrap().is_empty(), "no eviction yet");

    cache.push(TestEntry::new(ids_b)); // evicts A (RAM cap)
    let cap = captured.lock().unwrap();
    assert_eq!(cap.len(), 1, "exactly one spilled entry");
    assert_eq!(
        cap[0], expected_a,
        "spilled entry's block hash must match A"
    );
    assert_eq!(cache.stats.evictions, 1, "one eviction recorded");
}

/// (a)/slot-count variant + non-blocking proof: slot-count eviction
/// also spills, and `push` returns without the sink ever blocking. A
/// deliberately slow sink (sleeps) would surface here as a hang; we instead
/// assert push completes and the capture is synchronous-with-return only
/// because the *mock* is instant — the real sink is non-blocking by design
/// (see `spill::tests::push_does_not_block_on_slow_drain`).
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn spill_sink_captures_slot_count_eviction() {
    let n = 2 * BLOCK_TOKENS;
    let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut cache: PromptCache<TestEntry> =
        PromptCache::with_max_bytes(1, u64::MAX /* no RAM cap */);
    cache.set_spill_sink(Box::new(MockSink {
        captured: captured.clone(),
    }));

    let ids_a: Vec<u32> = (0..n as u32).map(|x| x * 2).collect();
    let ids_b: Vec<u32> = (0..n as u32).map(|x| x * 2 + 1).collect();
    let expected_a = *chained_block_hashes(&ids_a).last().unwrap();

    cache.push(TestEntry::new(ids_a));
    cache.push(TestEntry::new(ids_b)); // slot cap = 1 → evicts A
    let cap = captured.lock().unwrap();
    assert_eq!(cap.len(), 1, "slot-count eviction spills exactly one");
    assert_eq!(cap[0], expected_a);
}

/// OFF path: with no sink (`None`), eviction is the pre-pure
/// drop — nothing is captured, eviction counters are unchanged.
#[test]
fn no_spill_sink_is_pure_drop() {
    let n = 2 * BLOCK_TOKENS;
    let bytes_per_entry = n as u64 * 8;
    let max_bytes = bytes_per_entry + bytes_per_entry / 2;

    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(100, max_bytes);
    // No set_spill_sink call → spill is None.
    cache.push(entry(n));
    cache.push(entry(n)); // evicts first, pure drop
    assert_eq!(cache.slots.len(), 1, "RAM cap: one survives, OFF path");
    assert_eq!(cache.stats.evictions, 1, "eviction still counted");
}

// --- SSD-hydrate tests (generic, no disk, no model) ---

/// Mock SSD-hydrate source: returns a prebuilt `TestEntry` for any prompt
/// of at least one full block, else a miss. Records call count.
struct MockSource {
    entry_ids: Vec<u32>,
    calls: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl SsdHydrate<TestEntry> for MockSource {
    fn hydrate(
        &self,
        prompt_ids: &[u32],
        _seed: u64,
        _kv_quant: KvQuant,
        _policy: DispatchPolicy,
    ) -> Result<Option<TestEntry>> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if prompt_ids.len() < BLOCK_TOKENS {
            return Ok(None);
        }
        Ok(Some(TestEntry::new(self.entry_ids.clone())))
    }
}

/// (a)+(c): a RAM miss followed by `hydrate_from_ssd` promotes the
/// SSD entry into RAM, bumps `ssd_hits`, and a subsequent `find_best_prefix`
/// now serves it.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn ssd_hydrate_populates_ram_and_bumps_counter() {
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    cache.set_ssd_source(Box::new(MockSource {
        entry_ids: prompt.clone(),
        calls,
    }));

    // RAM miss (cache empty).
    assert!(
        cache.find_best_prefix(&prompt, FNV_OFFSET).is_none(),
        "cold RAM miss"
    );
    assert_eq!(cache.stats().ssd_hits, 0);

    // Hydrate from SSD → promoted into RAM, counter bumped.
    let slot = cache.hydrate_from_ssd(&prompt, FNV_OFFSET, TEST_QUANT, DispatchPolicy::default());
    assert!(slot.is_some(), "SSD hit must populate a RAM slot");
    assert_eq!(cache.stats().ssd_hits, 1, "ssd_hits must increment on hit");
    assert_eq!(cache.slots.len(), 1, "one slot now populated");

    // The hydrated entry is now served by find_best_prefix.
    let r = cache.find_best_prefix(&prompt, FNV_OFFSET);
    assert!(
        r.is_some(),
        "find_best_prefix must serve the hydrated entry"
    );
    let (_, blocks) = r.unwrap();
    assert_eq!(blocks, 2, "full 2-block prefix served from RAM");
}

/// A zero-slot cache never touches the SSD source.
///
/// Hydrating would read a `.kvb` and rebuild its K/V only for `push` to refuse
/// it — per request, for the life of the process — and the refusal would be
/// reported as a RAM-cap overflow, which is not the cause. The call-count
/// assertion is the load-bearing one: returning `None` after doing the work
/// would satisfy the other two.
#[test]
fn zero_slots_never_reads_the_ssd_source() {
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(0);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    cache.set_ssd_source(Box::new(MockSource {
        entry_ids: prompt.clone(),
        calls: std::sync::Arc::clone(&calls),
    }));

    assert!(
        cache
            .hydrate_from_ssd(&prompt, FNV_OFFSET, TEST_QUANT, DispatchPolicy::default())
            .is_none(),
        "a zero-slot cache cannot admit a hydrated entry"
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the source must not be queried at all — a disk read whose result can \
         only be discarded is pure waste"
    );
    assert_eq!(cache.stats().ssd_hits, 0);
    assert_eq!(cache.slots.len(), 0, "RAM untouched");
}

/// with no SSD source attached (`None`), `hydrate_from_ssd` is inert
/// and `ssd_hits` stays 0 — today's RAM-only behavior is unchanged.
#[test]
fn no_ssd_source_is_inert() {
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    assert!(cache.find_best_prefix(&prompt, FNV_OFFSET).is_none());
    assert!(
        cache
            .hydrate_from_ssd(&prompt, FNV_OFFSET, TEST_QUANT, DispatchPolicy::default())
            .is_none(),
        "no SSD source → always a miss"
    );
    assert_eq!(cache.stats().ssd_hits, 0, "ssd_hits stays 0 with no source");
    assert_eq!(cache.slots.len(), 0, "RAM untouched");
}

/// an SSD miss (source returns `Ok(None)`) leaves RAM + counters
/// untouched and returns `None` (caller falls through to prefill).
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn ssd_miss_leaves_ram_untouched() {
    // Sub-one-block prompt: mock returns a miss.
    let short: Vec<u32> = make_ids(BLOCK_TOKENS - 1);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    cache.set_ssd_source(Box::new(MockSource {
        entry_ids: short.clone(),
        calls: calls.clone(),
    }));
    assert!(
        cache
            .hydrate_from_ssd(&short, FNV_OFFSET, TEST_QUANT, DispatchPolicy::default())
            .is_none(),
        "SSD miss"
    );
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(cache.stats().ssd_hits, 0);
    assert_eq!(cache.slots.len(), 0);
}

/// over-cap admission guard applies to the SSD-hydrate path too: a
/// reconstructed block whose `kv_bytes()` alone exceeds the RAM cap is
/// refused admission by `push`, so `hydrate_from_ssd` must surface it as a
/// miss — NOT a served hit. `stats().ssd_hits` must NOT be bumped for an
/// entry that was never actually stored in RAM.
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn ssd_hydrate_over_cap_entry_is_not_counted_as_hit() {
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS); // kv_bytes() = 512 * 8 = 4096
                                                       // Cap strictly below the entry the mock source will reconstruct.
    let tiny_cap = (prompt.len() as u64 * 8) - 1;
    let mut cache: PromptCache<TestEntry> = PromptCache::with_max_bytes(4, tiny_cap);
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    cache.set_ssd_source(Box::new(MockSource {
        entry_ids: prompt.clone(),
        calls: calls.clone(),
    }));

    let slot = cache.hydrate_from_ssd(&prompt, FNV_OFFSET, TEST_QUANT, DispatchPolicy::default());
    assert!(
        slot.is_none(),
        "an over-cap reconstructed block must surface as a miss"
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "the source was queried exactly once"
    );
    assert_eq!(
        cache.stats().ssd_hits,
        0,
        "an over-cap hydrate must NOT be counted as a served ssd_hit"
    );
    assert_eq!(
        cache.slots.len(),
        0,
        "the over-cap block must not occupy a RAM slot"
    );
}

/// correctness: a partial hit truncates to exactly the matched block
/// boundary, so the re-prefilled tail starts at the correct absolute
/// position `matched_blocks * BLOCK_TOKENS`. A wrong truncation length here
/// would splice stale KV into the tail and silently corrupt output.
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
fn partial_hit_truncates_to_matched_block_boundary() {
    let cached: Vec<u32> = make_ids(4 * BLOCK_TOKENS); // 4 blocks
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(cached.clone()));

    // Request shares first 3 blocks, then diverges.
    let mut req = cached[..3 * BLOCK_TOKENS].to_vec();
    req.extend(50_000..50_000 + BLOCK_TOKENS as u32);
    let (idx, blocks) = cache
        .find_best_prefix(&req, FNV_OFFSET)
        .expect("3-block prefix hit");
    assert_eq!(blocks, 3, "exactly 3 leading blocks match");

    let mut cloned = cache.slots[idx].entry.deep_clone().unwrap();
    cloned
        .truncate_kv_to_block(blocks)
        .expect("a full-attention entry rolls back to a block boundary");
    assert_eq!(
        cloned.truncated_to.get(),
        Some(3 * BLOCK_TOKENS),
        "truncation must land at the matched-block boundary (3*256), not elsewhere"
    );
}

// ── resolve_ram_cap_bytes ────────────────────────────────────────

#[test]
fn ram_resolver_default_when_absent() {
    assert_eq!(resolve_ram_cap_bytes(None), DEFAULT_MAX_BYTES);
}

#[test]
fn ram_resolver_cli_only() {
    // 1.5 GiB
    let bytes = resolve_ram_cap_bytes(Some(1.5));
    assert_eq!(bytes, (1.5 * 1024.0 * 1024.0 * 1024.0) as u64);
}

#[test]
fn ram_resolver_cli_negative_falls_through_to_default() {
    // Negative CLI is rejected → default.
    let bytes = resolve_ram_cap_bytes(Some(-1.0));
    assert_eq!(bytes, DEFAULT_MAX_BYTES);
}

// ── unified ArchPromptCache ──────────────────────────────────────

/// unit test #1 (property): the chained block hash is computed by a
/// single function (`chained_block_hashes_seeded`) consumed by every arch.
/// This test pins the property that two arch entry types built from the
/// same `ids` + seed produce byte-identical digests — i.e. the unified
/// PromptCache's prefix-match path is arch-agnostic on the key side.
#[test]
fn chained_hash_arch_agnostic() {
    let ids: Vec<u32> = (0..(3 * BLOCK_TOKENS as u32)).collect();
    // Same seed → same digests for any caller (Gemma4, Qwen3, Qwen3.5-MoE).
    let a = chained_block_hashes_seeded(&ids, FNV_OFFSET);
    let b = chained_block_hashes_seeded(&ids, FNV_OFFSET);
    let c = chained_block_hashes_seeded(&ids, FNV_OFFSET);
    assert_eq!(
        a, b,
        "arch A vs arch B (same seed) must match byte-for-byte"
    );
    assert_eq!(
        b, c,
        "arch B vs arch C (same seed) must match byte-for-byte"
    );
    // And different seeds diverge (the layout-key salt actually
    // discriminates layouts).
    let salted = chained_block_hashes_seeded(&ids, FNV_OFFSET ^ 0xabcd_ef01_2345_6789);
    for (x, y) in a.iter().zip(salted.iter()) {
        assert_ne!(x, y, "different layout seeds must diverge");
    }
}

/// unit test #2: ReusePolicy is a hard runtime enum check.
/// An `ExactOnly` cache exposes the policy getter so the generate-loop
/// can route partial hits to Miss. This is the property test that the
/// policy field is observable + correct after construction.
#[test]
fn arch_prompt_cache_policy_is_observable() {
    let partial: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-partial", ReusePolicy::Partial, false);
    let exact: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-exact", ReusePolicy::ExactOnly, false);
    assert_eq!(partial.policy(), ReusePolicy::Partial);
    assert_eq!(exact.policy(), ReusePolicy::ExactOnly);
    assert_eq!(partial.arch_name(), "test-partial");
    assert_eq!(exact.arch_name(), "test-exact");
}

/// unit test #3: slot payload deep-clone semantics are preserved
/// after passing through `ArchPromptCache::with_inner_mut`. The closure
/// observes a `&mut Option<PromptCache<E>>` and any `push` / `find` is
/// reflected on subsequent calls. (Round-trip: build, push, find, clone,
/// verify identity.)
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn arch_cache_with_inner_mut_round_trip() {
    // No SsdSpiller/SsdHydrator impls for TestEntry, so we can't call
    // `ensure(...)`. Instead, install the cache manually inside the closure.
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-roundtrip", ReusePolicy::Partial, false);
    let ids: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut().unwrap().push(TestEntry::new(ids.clone()));
    });
    // Second-call observability: the pushed entry is reachable.
    let hit_blocks = arch.with_inner_mut(|g| {
        let cache = g.as_mut().unwrap();
        cache.find_best_prefix(&ids, FNV_OFFSET).map(|(_, b)| b)
    });
    assert_eq!(hit_blocks, Some(2), "pushed entry must hit on full prompt");
    // Deep-clone semantics: clone the slot and verify tokens identical.
    arch.with_inner_mut(|g| {
        let cache = g.as_ref().unwrap();
        let cloned = cache.slots[0].entry.deep_clone().unwrap();
        assert_eq!(cloned.prompt_token_ids(), &ids[..]);
    });
}

/// unit test #5: `read_cache_stats` returns `None` for an
/// uninitialised arch cache (no requests served yet). After installing a
/// fresh `PromptCache`, the stats are observable.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn arch_cache_stats_none_before_init() {
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-stats", ReusePolicy::Partial, false);
    assert!(
        arch.read_cache_stats().is_none(),
        "uninitialised arch cache must surface as None to /metrics/cache"
    );
    arch.with_inner_mut(|g| *g = Some(PromptCache::new(4)));
    let s = arch.read_cache_stats().expect("after init stats are Some");
    assert_eq!(s.hits, 0);
    assert_eq!(s.misses, 0);
}

/// unit test #6: an `ExactOnly` ArchPromptCache cannot accidentally
/// route a partial-prefix lookup to a non-Miss outcome — the policy is the
/// single source of truth. This test simulates the generate-loop gate by
/// checking that, on `ExactOnly`, a `find_best_prefix` partial hit (blocks
/// matched but tokens diverge) is treated as MUST-be-degraded by the
/// caller.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn exact_only_policy_forces_partial_match_to_miss_semantics() {
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-exact", ReusePolicy::ExactOnly, false);
    let cached: Vec<u32> = (0..(4 * BLOCK_TOKENS) as u32).collect();
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut().unwrap().push(TestEntry::new(cached.clone()));
    });
    // Build a divergent prompt: shares 3 blocks, then differs.
    let mut req = cached[..3 * BLOCK_TOKENS].to_vec();
    req.extend(50_000..50_000 + BLOCK_TOKENS as u32);

    // Generate-loop discipline (mirrored here): if policy == ExactOnly and
    // the full-token-equality check fails, the lookup MUST be Miss.
    let outcome = arch.with_inner_mut(|g| {
        let cache = g.as_mut().unwrap();
        let m = cache.find_best_prefix(&req, FNV_OFFSET);
        match m {
            Some((slot_idx, _b)) => {
                let exact = cache.slots[slot_idx].entry.prompt_token_ids() == req.as_slice();
                if exact || arch.policy() != ReusePolicy::ExactOnly {
                    "reuse" // would take Exact / Prefix path
                } else {
                    "miss" // ExactOnly + non-exact → forced Miss
                }
            }
            None => "miss",
        }
    });
    assert_eq!(
        outcome, "miss",
        "ExactOnly policy must force non-exact matches to Miss semantics"
    );
}

/// Issue #26 — codec-partitioned prefix-cache key (the anti-cross-serve guard).
///
/// A prefix cached under one KV codec must NOT be served to a request running a
/// different codec — the cached K/V bytes are codec-specific. The production
/// push/query seed is `FNV_OFFSET ^ layout_key ^ codec_salt(kv_quant)`; here we
/// store a slot under codec A's seed and assert that a query under codec A's
/// seed HITS (same codec → reuse) while a query under codec B's seed MISSES
/// (cross-codec → no serve), even though the token ids are byte-identical. This
/// is the namespacing that lets a resident model serve multiple codecs without
/// conflating their caches.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: PromptCache slots populated by construction just above each lookup"
)]
fn codec_partitioned_key_blocks_cross_codec_serve() {
    use rmlx_kv_quant::KvQuant;

    // Two distinct codecs → two distinct salts → two distinct seeds.
    let seed_a = FNV_OFFSET ^ KvQuant::None.cache_key_salt();
    let seed_b = FNV_OFFSET ^ KvQuant::K8V4.cache_key_salt();
    assert_ne!(seed_a, seed_b, "distinct codecs must yield distinct seeds");

    // Identical 2-block prompt for both codecs.
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);

    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    // Store the slot under codec A's seed (mirrors the production push).
    cache.push(TestEntry::new_seeded(prompt.clone(), seed_a));

    // 1. Same-codec query (seed A) → HIT (full 2-block prefix).
    let hit_a = cache.find_best_prefix(&prompt, seed_a);
    assert!(
        hit_a.is_some(),
        "same-codec query must hit the codec-A slot"
    );
    assert_eq!(hit_a.unwrap().1, 2, "full 2-block prefix must match");

    // 2. Cross-codec query (seed B) → MISS, even though the tokens are
    //    identical. The codec-B digest stream is disjoint from codec-A's, so
    //    the codec-A slot is never matched → no cross-serve of mismatched KV.
    let miss_b = cache.find_best_prefix(&prompt, seed_b);
    assert!(
        miss_b.is_none(),
        "cross-codec query must MISS the codec-A slot (anti-cross-serve guard)"
    );

    // 3. Coexistence: pushing the same prompt under codec B adds a *second*
    //    slot rather than colliding, and each codec now hits its own slot.
    cache.push(TestEntry::new_seeded(prompt.clone(), seed_b));
    assert_eq!(
        cache.slots.len(),
        2,
        "both codecs coexist as distinct slots"
    );
    assert!(
        cache.find_best_prefix(&prompt, seed_a).is_some(),
        "codec-A query still hits its own slot"
    );
    assert!(
        cache.find_best_prefix(&prompt, seed_b).is_some(),
        "codec-B query hits its own slot"
    );
}

// ── SSD hydrate seed symmetry ────────────────────────────────────────────────
//
// A hydrate source is handed the seed the RAM query is running under, and must
// build the promoted entry's `block_hashes` from *that value*. Any source that
// instead seeds from something it remembered — a codec or a model identity it
// captured when it was installed — returns an entry `find_best_prefix` cannot
// match. The block was read off disk, `ssd_hits` was incremented, an LRU slot
// was evicted to make room, and the request re-prefills anyway. Nothing errors.
//
// The source is installed once per architecture and outlives the model that
// installed it, so "something it remembered" is not hypothetical: it is
// whichever model attached last, at whatever codec the server launched with.

/// A mock `SsdHydrate<TestEntry>` that seeds from the passed-in `seed`, as the
/// production `SsdHydrator::lookup_seeded` does.
struct MockHydrateFromSeed {
    ids: Vec<u32>,
}

impl SsdHydrate<TestEntry> for MockHydrateFromSeed {
    fn hydrate(
        &self,
        _prompt_ids: &[u32],
        seed: u64,
        _kv_quant: KvQuant,
        _policy: DispatchPolicy,
    ) -> rmlx_core::error::Result<Option<TestEntry>> {
        let hashes = chained_block_hashes_seeded(&self.ids, seed);
        Ok(Some(TestEntry {
            ids: self.ids.clone(),
            hashes,
            truncated_to: std::cell::Cell::new(None),
            is_ssd_hydrated: true,
            kv_quant: None,
            hydrate_complete: true,
            reuse_kind: None,
        }))
    }
}

/// A mock that ignores the passed seed and rebuilds one from state captured at
/// construction — the defect shape. Stands in for a hydrator that remembered a
/// model signature or a codec from its own attach.
struct MockHydrateSelfSeeded {
    ids: Vec<u32>,
    stale_seed: u64,
}

impl SsdHydrate<TestEntry> for MockHydrateSelfSeeded {
    fn hydrate(
        &self,
        _prompt_ids: &[u32],
        _seed: u64,
        _kv_quant: KvQuant,
        _policy: DispatchPolicy,
    ) -> rmlx_core::error::Result<Option<TestEntry>> {
        let hashes = chained_block_hashes_seeded(&self.ids, self.stale_seed);
        Ok(Some(TestEntry {
            ids: self.ids.clone(),
            hashes,
            truncated_to: std::cell::Cell::new(None),
            is_ssd_hydrated: true,
            kv_quant: None,
            hydrate_complete: true,
            reuse_kind: None,
        }))
    }
}

/// A hydrated entry is findable exactly when its source used the query's seed.
///
/// 1. Source seeds from the passed value → the query that triggered the hydrate
///    finds the promoted entry (HIT).
/// 2. Source seeds from its own captured state → the same query cannot find it
///    (MISS). This is the failure being guarded against, asserted as a negative
///    so case 1 cannot pass by matching everything.
/// 3. A correctly hydrated entry is still not found by a different-codec query
///    — the seed's codec term keeps doing its job.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: values established by construction"
)]
fn hydrated_entry_is_findable_only_when_seeded_from_the_query() {
    // Non-zero layout_key to exercise the SSD-active code path.
    let layout_key: u64 = 0xdead_beef_0000_0001;
    let codec_a = KvQuant::K8V4;
    let codec_b = KvQuant::K8V8;
    assert_ne!(
        codec_a.cache_key_salt(),
        codec_b.cache_key_salt(),
        "codecs must have distinct salts"
    );

    // Two models of this arch; both seeds are what `consume` would compute.
    let seed_a = request_cache_seed(layout_key, codec_a, TEST_LAYERS, false, TEST_SIG);
    let seed_other_model = request_cache_seed(layout_key, codec_a, TEST_LAYERS, false, OTHER_SIG);

    let prompt_ids: Vec<u32> = make_ids(2 * BLOCK_TOKENS);

    // ── Case 1: source honours the passed seed → query HITs ─────────────────
    let mut cache_correct: PromptCache<TestEntry> = PromptCache::new(4);
    cache_correct.set_ssd_source(Box::new(MockHydrateFromSeed {
        ids: prompt_ids.clone(),
    }));
    assert!(
        cache_correct
            .find_best_prefix(&prompt_ids, seed_a)
            .is_none(),
        "RAM empty before hydrate"
    );
    let promoted =
        cache_correct.hydrate_from_ssd(&prompt_ids, seed_a, codec_a, DispatchPolicy::default());
    assert!(promoted.is_some(), "mock SSD source must hydrate");
    let after = cache_correct.find_best_prefix(&prompt_ids, seed_a);
    assert!(
        after.is_some(),
        "an entry seeded from the query's seed must be found by that query"
    );
    assert_eq!(after.unwrap().1, 2, "full 2-block prefix must match");

    // ── Case 2: source seeds from captured state → same query MISSes ────────
    // `stale_seed` is another resident model's seed: exactly what a hydrator
    // that remembered the last-attached model's signature would produce.
    let mut cache_broken: PromptCache<TestEntry> = PromptCache::new(4);
    cache_broken.set_ssd_source(Box::new(MockHydrateSelfSeeded {
        ids: prompt_ids.clone(),
        stale_seed: seed_other_model,
    }));
    cache_broken.hydrate_from_ssd(&prompt_ids, seed_a, codec_a, DispatchPolicy::default());
    assert!(
        cache_broken.find_best_prefix(&prompt_ids, seed_a).is_none(),
        "an entry seeded from the source's own state is unfindable by the query \
         that asked for it — the silent 0-hit"
    );

    // ── Case 3: correct hydrate under codec A → codec-B query MISSes ────────
    let seed_b = request_cache_seed(layout_key, codec_b, TEST_LAYERS, false, TEST_SIG);
    assert!(
        cache_correct
            .find_best_prefix(&prompt_ids, seed_b)
            .is_none(),
        "codec-A hydrated entry must not cross-serve a codec-B query"
    );
}

// ── Two models of one arch through the single per-arch SSD attach slot ───────
//
// `ArchPromptCache` holds ONE `Mutex<Option<AttachParams>>` and ONE hydrate
// source; `attach_ssd_tier` overwrites both wholesale, and the server can hold
// several models of an architecture resident at once (`--max-loaded-models`).
// So whichever model loads last owns the installed source, and every other
// resident model of that arch is served by it.
//
// That is only correct while the source carries no per-model state. These tests
// pin it: the identity comes from the request, so the shared source serves each
// model its own blocks and neither model's blocks to the other.

/// A stand-in for the on-disk tier: a content-addressed store keyed by the seed
/// a block was written under, which is exactly how `.kvb` rows behave (the
/// index key is the last chained digest, and the chain starts at the seed).
///
/// Holds no model identity and no codec of its own — it answers whatever seed
/// it is asked with, which is what makes it shareable across models.
///
/// Rows carry a block-aligned prefix of the prompt, strictly shorter than it,
/// as real spilled blocks do: the trailing partial block is never stored.
struct SeedKeyedStore {
    /// Seeds that have a block on "disk", with the token ids of each block.
    rows: Vec<(u64, Vec<u32>)>,
    /// Every seed this store was probed with, in order.
    probed: std::sync::Mutex<Vec<u64>>,
}

impl SsdHydrate<TestEntry> for SeedKeyedStore {
    fn hydrate(
        &self,
        _prompt_ids: &[u32],
        seed: u64,
        kv_quant: KvQuant,
        _policy: DispatchPolicy,
    ) -> rmlx_core::error::Result<Option<TestEntry>> {
        self.probed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(seed);
        let Some((_, ids)) = self.rows.iter().find(|(s, _)| *s == seed) else {
            return Ok(None); // no row under that seed — a true miss
        };
        Ok(Some(TestEntry {
            ids: ids.clone(),
            hashes: chained_block_hashes_seeded(ids, seed),
            truncated_to: std::cell::Cell::new(None),
            is_ssd_hydrated: true,
            kv_quant: Some(kv_quant),
            hydrate_complete: true,
            reuse_kind: Some(ReuseKind::StrictPrefix {
                prefix_len: ids.len(),
            }),
        }))
    }
}

/// Install `store` as the SSD source of a fresh cache on `arch`.
#[allow(
    clippy::unwrap_used,
    reason = "the cache is installed inside the same closure, immediately above the use"
)]
fn with_ssd_store(
    arch: &ArchPromptCache<TestEntry>,
    store: SeedKeyedStore,
) -> std::sync::Arc<std::sync::Mutex<Vec<u64>>> {
    let probed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut().unwrap().set_ssd_source(Box::new(store));
    });
    probed
}

/// Two models of one arch, one shared hydrate source: each hydrates its own
/// block and neither hydrates the other's.
///
/// Model A's block is the only one on disk. A must get it (or the tier is
/// silently dead for A), and B must not (or B decodes from A's K/V). Both go
/// through the same source instance, which is the only arrangement production
/// has — the per-arch attach slot holds one.
#[test]
fn two_models_of_one_arch_share_a_source_and_not_each_others_blocks() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    // The block-aligned prefix a real spill would hold: `make_ids` is a
    // prefix-stable range, so this is exactly `prompt`'s first block.
    let stored = make_ids(BLOCK_TOKENS);

    // `active_layout_key()` is 0 here (no attach recorded), so these are the
    // seeds `consume` will compute for the two models.
    let seed_a = request_cache_seed(0, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
    let seed_b = request_cache_seed(0, TEST_QUANT, TEST_LAYERS, false, OTHER_SIG);
    assert_ne!(seed_a, seed_b, "two models must not share a seed");

    // ── Model A: its block is on disk under its own seed → hydrate. ──────────
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-two-models-ssd", ReusePolicy::Partial, false);
    with_ssd_store(
        &arch,
        SeedKeyedStore {
            rows: vec![(seed_a, stored.clone())],
            probed: std::sync::Mutex::new(Vec::new()),
        },
    );
    let _ = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
    assert_eq!(
        arch.read_cache_stats().map(|s| s.ssd_hits),
        Some(1),
        "the model whose block is on disk must be served from the SSD tier"
    );

    // ── Model B: same source, same prompt, same codec, different model. ──────
    let arch_b: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-two-models-ssd", ReusePolicy::Partial, false);
    with_ssd_store(
        &arch_b,
        SeedKeyedStore {
            rows: vec![(seed_a, stored)],
            probed: std::sync::Mutex::new(Vec::new()),
        },
    );
    let consumed_b = arch_b.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, OTHER_SIG);
    assert_eq!(
        arch_b.read_cache_stats().map(|s| s.ssd_hits),
        Some(0),
        "another model of the same arch must not be served A's block"
    );
    assert_eq!(
        tag(&consumed_b),
        ConsumedTag::Miss,
        "B re-prefills rather than decoding from A's K/V"
    );
}

/// Both models resident against one source that holds *both* their blocks:
/// each is served its own, and the tier is live for both.
///
/// This is the half a snapshotted-at-attach identity fails. With the source
/// pinned to whichever model attached last, one of these two would report
/// `ssd_hits == 0` for a block that is sitting on disk.
#[test]
fn each_resident_model_hydrates_its_own_block_from_the_shared_source() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    // The block-aligned prefix a real spill would hold: `make_ids` is a
    // prefix-stable range, so this is exactly `prompt`'s first block.
    let stored = make_ids(BLOCK_TOKENS);
    let seed_a = request_cache_seed(0, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
    let seed_b = request_cache_seed(0, TEST_QUANT, TEST_LAYERS, false, OTHER_SIG);

    for (sig, label) in [(TEST_SIG, "model A"), (OTHER_SIG, "model B")] {
        let arch: ArchPromptCache<TestEntry> =
            ArchPromptCache::new("test-both-resident", ReusePolicy::Partial, false);
        // Both models' blocks are on disk, as they would be after both have
        // served the prompt once and been evicted from RAM.
        with_ssd_store(
            &arch,
            SeedKeyedStore {
                rows: vec![(seed_a, stored.clone()), (seed_b, stored.clone())],
                probed: std::sync::Mutex::new(Vec::new()),
            },
        );
        let consumed = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, sig);
        assert_eq!(
            arch.read_cache_stats().map(|s| s.ssd_hits),
            Some(1),
            "{label} must hydrate its own block from the shared source"
        );
        // A hydrate the follow-up lookup cannot match is a miss that got
        // counted as a hit: the block was read, an LRU slot was evicted for it,
        // and the request re-prefills anyway. Assert the promoted entry is
        // actually served, not merely that the counter moved.
        assert_ne!(
            tag(&consumed),
            ConsumedTag::Miss,
            "{label}'s hydrated block must be matched by the query that asked \
             for it, not counted as a hit and then re-prefilled"
        );
    }
}

/// The request's codec reaches the source, not the launch codec.
///
/// A hot-swapped request (`kv_quant` in the request body) probes under its own
/// codec's seed and tags the reconstructed entry with it. A source that
/// substituted an attach-time codec would probe the wrong digest stream, and
/// then tag the entry so the quant guard evicts it.
#[test]
fn hydrate_probes_under_the_requests_codec_not_the_launch_codec() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    // The block-aligned prefix a real spill would hold: `make_ids` is a
    // prefix-stable range, so this is exactly `prompt`'s first block.
    let stored = make_ids(BLOCK_TOKENS);
    let launch = KvQuant::K8V8;
    let request = KvQuant::K8V4;
    assert_ne!(launch, request);

    // Only the hot-swapped codec's block is on disk.
    let seed_request = request_cache_seed(0, request, TEST_LAYERS, false, TEST_SIG);

    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-codec-hotswap", ReusePolicy::Partial, false);
    with_ssd_store(
        &arch,
        SeedKeyedStore {
            rows: vec![(seed_request, stored.clone())],
            probed: std::sync::Mutex::new(Vec::new()),
        },
    );
    let _ = arch.consume(&prompt, request, TEST_LAYERS, false, TEST_SIG);
    assert_eq!(
        arch.read_cache_stats().map(|s| s.ssd_hits),
        Some(1),
        "a request at the hot-swapped codec must find the block it stored"
    );

    // And the launch codec must not reach that block.
    let arch2: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-codec-hotswap", ReusePolicy::Partial, false);
    with_ssd_store(
        &arch2,
        SeedKeyedStore {
            rows: vec![(seed_request, stored)],
            probed: std::sync::Mutex::new(Vec::new()),
        },
    );
    let _ = arch2.consume(&prompt, launch, TEST_LAYERS, false, TEST_SIG);
    assert_eq!(
        arch2.read_cache_stats().map(|s| s.ssd_hits),
        Some(0),
        "a request at another codec must not be served that block"
    );
}

// ── consume() engine — golden discriminant, policy×entry matrix, edges ───────
//
// These exercise the shared `ArchPromptCache::consume` decision over the
// `TestEntry` mock. The cache is installed via `with_inner_mut` (TestEntry
// lacks the SsdSpiller/SsdHydrator bounds `ensure` needs). On a RAM-only run
// `active_layout_key() == 0`, so the engine's seed is `FNV_OFFSET ^
// kv_quant.cache_key_salt()` — `TestEntry::for_quant` salts its block hashes to
// match, and tags the entry with that same quant so the quant-guard accepts it.

/// The runtime KV quant used for every consume test (entries are salted +
/// tagged to match via `TestEntry::for_quant`).
const TEST_QUANT: KvQuant = KvQuant::K8V8;

/// Model signature every consume test pushes and queries under. `TestEntry::
/// for_quant` seeds its digests with it, so a query carrying a *different*
/// signature must not match — see `consume_other_model_sig_is_miss`.
const TEST_SIG: u64 = 0x1111_2222_3333_4444;
/// Decoder-layer count every seed in this file is computed at. The seed folds
/// the per-layer codec mixture, so a push and its query have to agree on it —
/// exactly as production does, where it is the model's own layer count.
const TEST_LAYERS: usize = 36;

/// A second model's signature. Same arch, same static cache, different model.
const OTHER_SIG: u64 = 0x5555_6666_7777_8888;

/// Compact discriminant of a `Consumed` for golden-pair assertions.
#[derive(Debug, PartialEq, Eq)]
enum ConsumedTag {
    Exact,
    Reuse(ReuseKind),
    Miss,
}

fn tag<E>(c: &Consumed<E>) -> ConsumedTag {
    match c {
        Consumed::Exact(_) => ConsumedTag::Exact,
        Consumed::Reuse { kind, .. } => ConsumedTag::Reuse(*kind),
        Consumed::Miss => ConsumedTag::Miss,
    }
}

/// Install a single pushed entry into a fresh `ArchPromptCache` and run consume.
#[allow(
    clippy::unwrap_used,
    reason = "test-only: the cache is installed inside the same closure just above each use"
)]
fn consume_one(
    policy: ReusePolicy,
    pushed: TestEntry,
    prompt_ids: &[u32],
    has_image: bool,
) -> (ArchPromptCache<TestEntry>, Consumed<TestEntry>) {
    let arch: ArchPromptCache<TestEntry> = ArchPromptCache::new("test", policy, false);
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut().unwrap().push(pushed);
    });
    let out = arch.consume(prompt_ids, TEST_QUANT, TEST_LAYERS, has_image, TEST_SIG);
    (arch, out)
}

/// A second model of the same architecture must not be served another model's
/// K/V, even when the prompts are token-for-token identical.
///
/// The prompt cache is one static per arch, so both models share it, and the
/// `Exact` arm's token-id equality check cannot separate them — the tokens
/// really are equal. Only the model term in the key can. Serving the hit would
/// replay model A's K/V and A's first decode token through model B's weights:
/// wrong output, no error.
///
/// Paired with `consume_exact_ram_entry` directly above, which is the same
/// setup at the *same* signature and must still hit — otherwise this test
/// passes for the trivial reason that nothing ever matches.
#[test]
fn consume_other_model_sig_is_miss() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-two-models", ReusePolicy::Partial, false);
    arch.ensure(4);
    arch.with_inner_mut(|g| {
        if let Some(cache) = g.as_mut() {
            cache.push(TestEntry::for_quant(prompt.clone(), TEST_QUANT));
        }
    });

    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)),
        ConsumedTag::Exact,
        "the model that stored the slot must still be served from it"
    );
    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, OTHER_SIG)),
        ConsumedTag::Miss,
        "a different model of the same arch must re-prefill, not decode from \
         another model's K/V"
    );
}

/// RAM exact hit: stored ids == request ids, not hydrated → `Exact`.
#[test]
fn consume_exact_ram_entry() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    let (_arch, out) = consume_one(
        ReusePolicy::ExactOnly,
        TestEntry::for_quant(prompt.clone(), TEST_QUANT),
        &prompt,
        false,
    );
    assert_eq!(tag(&out), ConsumedTag::Exact);
}

/// Hydrated entry whose ids exactly equal the request (block-aligned, no tail)
/// must be `Miss` — the Exact arm excludes hydrated entries (placeholder
/// first_id) and the reuse hook's strict-less guard excludes equal-length.
#[test]
fn consume_hydrated_equal_length_is_miss() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    // Moe-style hook would only return StrictPrefix when stored.len() <
    // prompt.len(); equal-length yields None here regardless. Configure a
    // StrictPrefix reuse_kind to prove even a permissive hook can't rescue the
    // equal-length hydrated entry (the engine never reaches reuse for it).
    let entry = TestEntry::for_quant(prompt.clone(), TEST_QUANT)
        .with_ssd_hydrated(true)
        .with_reuse_kind(ReuseKind::StrictPrefix {
            prefix_len: prompt.len(),
        });
    let (_arch, out) = consume_one(ReusePolicy::ExactOnly, entry, &prompt, false);
    assert_eq!(tag(&out), ConsumedTag::Miss);
}

/// ExactOnly forbids a fresh (non-hydrated) partial match → `Miss`, even if the
/// hook would otherwise offer a StrictPrefix (the gate never calls it).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test-only: slice bound is a literal block multiple <= the just-built prompt length"
)]
fn consume_exactonly_fresh_partial_is_miss() {
    let cached = make_ids(4 * BLOCK_TOKENS);
    // Request shares the first 3 blocks then diverges.
    let mut req = cached[..3 * BLOCK_TOKENS].to_vec();
    req.extend(50_000..50_000 + BLOCK_TOKENS as u32);
    let entry = TestEntry::for_quant(cached, TEST_QUANT).with_reuse_kind(ReuseKind::StrictPrefix {
        prefix_len: 3 * BLOCK_TOKENS,
    });
    let (_arch, out) = consume_one(ReusePolicy::ExactOnly, entry, &req, false);
    assert_eq!(tag(&out), ConsumedTag::Miss);
}

/// ExactOnly STILL permits a hydrated strict-prefix (the moe HydratedTail) →
/// `Reuse{StrictPrefix}`.
#[test]
fn consume_exactonly_hydrated_strict_prefix_is_reuse() {
    let stored = make_ids(2 * BLOCK_TOKENS);
    let mut req = stored.clone();
    req.extend(70_000..70_000 + BLOCK_TOKENS as u32);
    let entry = TestEntry::for_quant(stored.clone(), TEST_QUANT)
        .with_ssd_hydrated(true)
        .with_reuse_kind(ReuseKind::StrictPrefix {
            prefix_len: stored.len(),
        });
    let (_arch, out) = consume_one(ReusePolicy::ExactOnly, entry, &req, false);
    assert_eq!(
        tag(&out),
        ConsumedTag::Reuse(ReuseKind::StrictPrefix {
            prefix_len: 2 * BLOCK_TOKENS,
        })
    );
    // The Reuse carries the cloned entry (its prefix tokens), which the arch
    // forwards a tail on top of. Assert the clone preserved the stored prefix.
    if let Consumed::Reuse { entry, .. } = out {
        assert_eq!(
            entry.prompt_token_ids(),
            stored.as_slice(),
            "the reused clone must carry the stored prefix tokens"
        );
    } else {
        panic!("asserted Reuse above");
    }
}

/// Partial policy permits a fresh (non-hydrated) prefix reuse via the hook.
/// Here the hook returns BlockTruncate → `Reuse{BlockTruncate}`.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test-only: slice bound is a literal block multiple <= the just-built prompt length"
)]
fn consume_partial_fresh_prefix_is_reuse() {
    let cached = make_ids(4 * BLOCK_TOKENS);
    let mut req = cached[..3 * BLOCK_TOKENS].to_vec();
    req.extend(80_000..80_000 + BLOCK_TOKENS as u32);
    let entry =
        TestEntry::for_quant(cached, TEST_QUANT).with_reuse_kind(ReuseKind::BlockTruncate {
            effective_blocks: 3,
        });
    let (_arch, out) = consume_one(ReusePolicy::Partial, entry, &req, false);
    assert_eq!(
        tag(&out),
        ConsumedTag::Reuse(ReuseKind::BlockTruncate {
            effective_blocks: 3
        })
    );
}

/// Partial policy with an incomplete hydrated entry (gemma4 SWA payload-less
/// layer) → `Miss` via `is_hydrate_complete == false`, even though the hook
/// would offer a StrictPrefix.
#[test]
fn consume_incomplete_hydrate_partial_is_miss() {
    let stored = make_ids(2 * BLOCK_TOKENS);
    let mut req = stored.clone();
    req.extend(90_000..90_000 + BLOCK_TOKENS as u32);
    let entry = TestEntry::for_quant(stored.clone(), TEST_QUANT)
        .with_ssd_hydrated(true)
        .with_hydrate_complete(false)
        .with_reuse_kind(ReuseKind::StrictPrefix {
            prefix_len: stored.len(),
        });
    let (_arch, out) = consume_one(ReusePolicy::Partial, entry, &req, false);
    assert_eq!(tag(&out), ConsumedTag::Miss);
}

/// `has_image == true` → `Miss` WITHOUT touching the cache: a pre-existing exact
/// entry must survive (not evicted), so a later text request still hits it.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: cache installed in the same closure just above"
)]
fn consume_has_image_is_miss_no_cache_touch() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test", ReusePolicy::ExactOnly, false);
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut()
            .unwrap()
            .push(TestEntry::for_quant(prompt.clone(), TEST_QUANT));
    });

    // Image request: Miss, and the cache is not consulted/evicted.
    let img = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, true, TEST_SIG);
    assert_eq!(tag(&img), ConsumedTag::Miss);
    let slots = arch.with_inner_mut(|g| g.as_ref().unwrap().slots.len());
    assert_eq!(slots, 1, "image request must not evict the existing entry");

    // The surviving entry is still served as Exact to a text request.
    let txt = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
    assert_eq!(tag(&txt), ConsumedTag::Exact);
}

/// Sub-one-block prompt → always `Miss` (find_best_prefix needs ≥1 full block).
#[test]
fn consume_short_prompt_under_one_block_is_miss() {
    let short = make_ids(BLOCK_TOKENS - 1);
    let (_arch, out) = consume_one(
        ReusePolicy::ExactOnly,
        TestEntry::for_quant(short.clone(), TEST_QUANT),
        &short,
        false,
    );
    assert_eq!(tag(&out), ConsumedTag::Miss);
}

/// Guard contract: consume is never called for an empty prompt (the arch
/// early-returns on `n_tokens == 0` before `ensure_prompt_cache`). We still pin
/// that an empty prompt yields `Miss` if it ever reaches the engine, so the
/// guard is the only place that decides "no generation."
#[test]
fn consume_empty_after_guard() {
    let empty: Vec<u32> = Vec::new();
    let (_arch, out) = consume_one(
        ReusePolicy::ExactOnly,
        TestEntry::for_quant(make_ids(2 * BLOCK_TOKENS), TEST_QUANT),
        &empty,
        false,
    );
    assert_eq!(tag(&out), ConsumedTag::Miss);
}

/// Quant-mismatch degrade: the stored entry's quant differs from the runtime
/// quant → evict + `Miss`. The entry is salted for `K8V4` but queried at the
/// `TEST_QUANT` (`K8V8`) seed, so it would already miss the block-hash match;
/// to isolate the quant-guard we salt with the RUNTIME seed but tag a different
/// stored quant so the match fires and the guard rejects it.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: cache installed in the same closure just above"
)]
fn consume_quant_mismatch_evicts_and_misses() {
    let prompt = make_ids(2 * BLOCK_TOKENS);
    // Salt with the runtime seed so find_best_prefix matches, but tag a DIFFERENT
    // stored quant so the quant-guard rejects it.
    let runtime_seed = request_cache_seed(0, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
    let mut entry = TestEntry::new_seeded(prompt.clone(), runtime_seed);
    entry.kv_quant = Some(KvQuant::K8V4); // != TEST_QUANT (K8V8)

    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test", ReusePolicy::ExactOnly, false);
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut().unwrap().push(entry);
    });
    let out = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
    assert_eq!(tag(&out), ConsumedTag::Miss);
    let slots = arch.with_inner_mut(|g| g.as_ref().unwrap().slots.len());
    assert_eq!(slots, 0, "quant-mismatch must evict the unusable slot");
}

// ── degrade-trace coverage ───────────────────────────────────────────────────
//
// Every consume degrade branch must emit exactly one `debug!{branch=...}` so a
// silent-drop regression (the qwen3/moe arms were silent before unification)
// fails this test rather than passing every other one.
//
// Capturing tracing events reliably from a test that shares one binary with
// many other tracing-emitting tests is subtle: the per-callsite Interest cache
// is GLOBAL and is populated by whatever dispatcher was current the FIRST time a
// callsite fired. A thread-local `with_default` subscriber cannot retroactively
// un-disable a callsite that a sibling test already cached as `never` under the
// no-op default. The fix is a process-global "tee" subscriber installed once:
// its `register_callsite` returns `always` (so callsites are never cached
// disabled) and its `event` forwards the `branch` field to a THREAD-LOCAL sink
// that is only armed inside `capture_branches`. Outside a capture it is inert.

thread_local! {
    static BRANCH_SINK: std::cell::RefCell<Option<Vec<String>>> =
        const { std::cell::RefCell::new(None) };
}

struct BranchVisitor<'a> {
    out: &'a mut Vec<String>,
}

impl tracing::field::Visit for BranchVisitor<'_> {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "branch" {
            self.out.push(value.to_owned());
        }
    }
    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

/// Process-global tee subscriber. Always-interest so no callsite is cached as
/// disabled; forwards `branch` fields to the thread-local sink when armed.
struct TeeBranchSubscriber;

impl tracing::Subscriber for TeeBranchSubscriber {
    fn register_callsite(
        &self,
        _meta: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::always()
    }
    fn enabled(&self, _md: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn new_span(&self, _span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        BRANCH_SINK.with(|sink| {
            if let Some(out) = sink.borrow_mut().as_mut() {
                let mut v = BranchVisitor { out };
                event.record(&mut v);
            }
        });
    }
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

/// Run `f` with the thread-local branch sink armed; return captured branches.
///
/// Installs the process-global tee subscriber on first use. Because the tee's
/// `register_callsite` is `always`, the consume callsites dispatch to it on
/// every thread regardless of which test fired them first — only the armed
/// thread records, so the capture is deterministic across the shared binary.
///
/// test-ordering note: `set_global_default` is called exactly once (via
/// `Once`) and is never uninstalled. No other test in this binary may rely on
/// the global tracing dispatcher's span-filter or subscriber-specific
/// semantics — the permanent install here will shadow any later attempt to set
/// a different global default. Tests that need custom tracing must use a
/// per-thread or scoped subscriber rather than the global dispatcher.
fn capture_branches(f: impl FnOnce()) -> Vec<String> {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // Ignore an error: another test having set a global default would only
        // mean our tee is not installed; the assertions below would then catch
        // a silent-drop, which is the desired failure mode.
        let _ = tracing::subscriber::set_global_default(TeeBranchSubscriber);
    });
    BRANCH_SINK.with(|sink| *sink.borrow_mut() = Some(Vec::new()));
    f();
    BRANCH_SINK.with(|sink| sink.borrow_mut().take().unwrap_or_default())
}

/// Each degrade path emits exactly one `debug!{branch=...}` with the expected
/// branch name. All six scenarios run inside ONE capture scope (one subscriber
/// install + interest rebuild) so the collected `branch` sequence is
/// deterministic — the per-call capture variant raced the global tracing
/// interest cache against other tests sharing the binary.
///
/// Covers: has_image, quant_mismatch, incomplete_hydrate, non_reusable (fresh
/// partial under Partial), hydrated_declined_to_exact (fresh partial under
/// ExactOnly), and the hydrated equal-length case (declines via non_reusable).
/// Exactly one `branch` per consume call.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: caches installed in the same closures just above each consume"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "test-only: slice bound is a literal block multiple <= the just-built prompt length"
)]
fn consume_degrade_branches_each_emit_one_debug() {
    let prompt = make_ids(2 * BLOCK_TOKENS);

    let branches = capture_branches(|| {
        // 1. has_image → "has_image" (no cache touch).
        {
            let arch: ArchPromptCache<TestEntry> =
                ArchPromptCache::new("test", ReusePolicy::ExactOnly, false);
            arch.with_inner_mut(|g| *g = Some(PromptCache::new(4)));
            let _ = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, true, TEST_SIG);
        }
        // 2. quant_mismatch → "quant_mismatch".
        {
            let runtime_seed = request_cache_seed(0, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
            let mut entry = TestEntry::new_seeded(prompt.clone(), runtime_seed);
            entry.kv_quant = Some(KvQuant::K8V4);
            let arch: ArchPromptCache<TestEntry> =
                ArchPromptCache::new("test", ReusePolicy::ExactOnly, false);
            arch.with_inner_mut(|g| {
                *g = Some(PromptCache::new(4));
                g.as_mut().unwrap().push(entry);
            });
            let _ = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
        }
        // 3. incomplete_hydrate (Partial, hydrated, incomplete) → "incomplete_hydrate".
        {
            let mut req = prompt.clone();
            req.extend(11_000..11_000 + BLOCK_TOKENS as u32);
            let entry = TestEntry::for_quant(prompt.clone(), TEST_QUANT)
                .with_ssd_hydrated(true)
                .with_hydrate_complete(false)
                .with_reuse_kind(ReuseKind::StrictPrefix {
                    prefix_len: prompt.len(),
                });
            let arch: ArchPromptCache<TestEntry> =
                ArchPromptCache::new("test", ReusePolicy::Partial, false);
            arch.with_inner_mut(|g| {
                *g = Some(PromptCache::new(4));
                g.as_mut().unwrap().push(entry);
            });
            let _ = arch.consume(&req, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
        }
        // 4. non_reusable (Partial, fresh partial, hook returns None) → "non_reusable".
        {
            let cached = make_ids(4 * BLOCK_TOKENS);
            let mut req = cached[..3 * BLOCK_TOKENS].to_vec();
            req.extend(60_000..60_000 + BLOCK_TOKENS as u32);
            let entry = TestEntry::for_quant(cached, TEST_QUANT);
            let arch: ArchPromptCache<TestEntry> =
                ArchPromptCache::new("test", ReusePolicy::Partial, false);
            arch.with_inner_mut(|g| {
                *g = Some(PromptCache::new(4));
                g.as_mut().unwrap().push(entry);
            });
            let _ = arch.consume(&req, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
        }
        // 5. hydrated_declined_to_exact (ExactOnly, fresh partial, not token-equal,
        //    reuse not permitted) → "hydrated_declined_to_exact".
        {
            let cached = make_ids(4 * BLOCK_TOKENS);
            let mut req = cached[..3 * BLOCK_TOKENS].to_vec();
            req.extend(61_000..61_000 + BLOCK_TOKENS as u32);
            let entry = TestEntry::for_quant(cached, TEST_QUANT);
            let arch: ArchPromptCache<TestEntry> =
                ArchPromptCache::new("test", ReusePolicy::ExactOnly, false);
            arch.with_inner_mut(|g| {
                *g = Some(PromptCache::new(4));
                g.as_mut().unwrap().push(entry);
            });
            let _ = arch.consume(&req, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
        }
        // 6. hydrated equal-length: reuse-eligible (complete) but the hook
        //    declines (strict-less) → "non_reusable".
        {
            let entry = TestEntry::for_quant(prompt.clone(), TEST_QUANT)
                .with_ssd_hydrated(true)
                .with_reuse_kind(ReuseKind::StrictPrefix {
                    prefix_len: prompt.len(),
                });
            let arch: ArchPromptCache<TestEntry> =
                ArchPromptCache::new("test", ReusePolicy::ExactOnly, false);
            arch.with_inner_mut(|g| {
                *g = Some(PromptCache::new(4));
                g.as_mut().unwrap().push(entry);
            });
            let _ = arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG);
        }
    });

    // Exactly one branch per consume call, in order.
    assert_eq!(
        branches,
        vec![
            "has_image".to_owned(),
            "quant_mismatch".to_owned(),
            "incomplete_hydrate".to_owned(),
            "non_reusable".to_owned(),
            "hydrated_declined_to_exact".to_owned(),
            "non_reusable".to_owned(),
        ],
        "each consume degrade branch must emit exactly one debug!{{branch=...}}"
    );
}

// ── ensure() + the zero-slot cache ──────────────────────────────────────────
//
// `ensure` runs once per generation on every arch, so its "already the right
// capacity" arm has to be reachable for every value it is called with. When it
// is not, the cache is rebuilt per generation: snapshots are discarded, the
// hit/miss counters restart at zero, and the SSD sinks are re-installed — which
// looks like "caching is off" from the outside while in fact building and
// throwing away a cache each time. The counters are the sharp edge: a caller
// reading cache activity as `after - before` around a generation then reads
// `0 - 0` and cannot tell that from "nothing happened".
//
// The tests below need `ensure`, which is bounded on `SsdHydrator:
// SsdHydrate<E>`. The SSD tier is never attached here, so this impl exists only
// to satisfy that bound and is never called.
impl SsdHydrate<TestEntry> for SsdHydrator {
    fn hydrate(
        &self,
        _prompt_ids: &[u32],
        _seed: u64,
        _kv_quant: KvQuant,
        _policy: DispatchPolicy,
    ) -> Result<Option<TestEntry>> {
        Ok(None)
    }
}

/// Cumulative miss count observable across an arch cache, or 0 before the first
/// `ensure` builds it.
fn misses(arch: &ArchPromptCache<TestEntry>) -> u64 {
    arch.read_cache_stats().map_or(0, |s| s.misses)
}

/// Two zero-slot generations must leave a cumulative miss count of 2.
///
/// Each generation calls `ensure(0)` and then consults the cache once. Reading
/// `1` means the second `ensure(0)` rebuilt the cache and reset the counters, so
/// the first generation's miss is gone.
#[test]
fn zero_slots_accumulates_misses_across_generations() {
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-zero-slots", ReusePolicy::Partial, false);
    let prompt = make_ids(2 * BLOCK_TOKENS);

    arch.ensure(0);
    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)),
        ConsumedTag::Miss
    );
    assert_eq!(misses(&arch), 1, "first generation must record its miss");

    arch.ensure(0);
    assert_eq!(
        misses(&arch),
        1,
        "ensure(0) on a cache already built with 0 slots must be a no-op — a rebuild \
         here resets the counters and erases the first generation's miss"
    );

    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)),
        ConsumedTag::Miss
    );
    assert_eq!(
        misses(&arch),
        2,
        "two zero-slot generations are two misses; 1 means the counters restarted"
    );
}

/// Zero slots is a real disabled state, not a one-slot cache.
///
/// A snapshot offered to it is refused, so a repeat of the identical prompt
/// cannot be served from RAM and re-prefills — which is what an operator asking
/// for zero slots is asking for.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: `ensure` on the line above installs the cache the closure unwraps"
)]
fn zero_slots_stores_nothing() {
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-zero-store", ReusePolicy::Partial, false);
    let prompt = make_ids(2 * BLOCK_TOKENS);

    arch.ensure(0);
    let stored = arch.with_inner_mut(|g| {
        g.as_mut()
            .unwrap()
            .push(TestEntry::for_quant(prompt.clone(), TEST_QUANT))
    });
    assert_eq!(
        stored, None,
        "a zero-slot cache must refuse admission rather than store into a clamped slot"
    );

    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)),
        ConsumedTag::Miss,
        "an identical repeat must still miss — a zero-slot cache serves nothing"
    );
}

/// `ensure` still rebuilds when the capacity genuinely changes, and still does
/// not when it has not. Without this the repair could be "never rebuild", which
/// would silently ignore a capacity change between model loads.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: `ensure` on the line above installs the cache the closure unwraps"
)]
fn ensure_rebuilds_only_on_a_real_capacity_change() {
    let arch: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-ensure-rebuild", ReusePolicy::Partial, false);
    let prompt = make_ids(2 * BLOCK_TOKENS);

    arch.ensure(4);
    arch.with_inner_mut(|g| {
        g.as_mut()
            .unwrap()
            .push(TestEntry::for_quant(prompt.clone(), TEST_QUANT));
    });

    arch.ensure(4);
    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)),
        ConsumedTag::Exact,
        "the same capacity must keep the stored snapshot"
    );

    arch.ensure(2);
    assert_eq!(
        tag(&arch.consume(&prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)),
        ConsumedTag::Miss,
        "a changed capacity must rebuild the cache and discard its snapshots"
    );
}
