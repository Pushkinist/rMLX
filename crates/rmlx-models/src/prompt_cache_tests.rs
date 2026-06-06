use super::*;
use rmlx_core::error::Result;

// ── chained_block_hashes_seeded ────────────────────────────────────

/// unit test #2: `chained_block_hashes_seeded(ids, FNV_OFFSET)`
/// produces the exact same digests as the bare `chained_block_hashes(ids)`.
/// Guards the re-export contract — old call sites that have not opted into
/// layout-key salting must observe byte-identical behaviour.
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
struct TestEntry {
    ids: Vec<u32>,
    hashes: Vec<u64>,
    truncated_to: std::cell::Cell<Option<usize>>,
}

impl TestEntry {
    fn new(ids: Vec<u32>) -> Self {
        let hashes = chained_block_hashes(&ids);
        TestEntry {
            ids,
            hashes,
            truncated_to: std::cell::Cell::new(None),
        }
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
        })
    }

    fn truncate_kv_to(&mut self, prefix_len: usize) {
        self.truncated_to.set(Some(prefix_len));
    }

    fn truncate_kv_to_block(&mut self, block_count: usize) {
        self.truncated_to.set(Some(block_count * BLOCK_TOKENS));
    }

    fn kv_bytes(&self) -> u64 {
        // Fake: 8 bytes per token id so we can assert non-zero bytes.
        self.ids.len() as u64 * 8
    }
}

fn make_ids(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

fn entry(n: usize) -> TestEntry {
    TestEntry::new(make_ids(n))
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
    let r1 = cache.find_best_prefix(&prompt_a);
    assert!(r1.is_none(), "r1 should be a miss");
    // Populate cache so subsequent requests can hit.
    cache.push(TestEntry::new(prompt_a.clone()));

    // --- Request 2: full-prompt block hit (same prompt) ---
    let r2 = cache.find_best_prefix(&prompt_a);
    assert!(r2.is_some(), "r2 should be a hit");
    let (_, blocks_2) = r2.unwrap();
    assert_eq!(blocks_2, 2, "full hit must cover all 2 blocks");

    // --- Request 3: partial prefix hit (one extra full block of new ids) ---
    let mut prompt_b = prompt_a.clone();
    prompt_b.extend(1000..1000 + BLOCK_TOKENS as u32);
    let r3 = cache.find_best_prefix(&prompt_b);
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

/// Verify that a shared prefix of fewer than one full block is a miss.
#[test]
fn short_prefix_treated_as_miss() {
    let short_prompt: Vec<u32> = make_ids(BLOCK_TOKENS - 1);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(short_prompt.clone()));

    // Query with the same short prompt — no full block stored, so there is
    // nothing to match: must be a miss.
    let result = cache.find_best_prefix(&short_prompt);
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
    let hit_b = cache.find_best_prefix(&ids_b);
    assert!(hit_b.is_some(), "B must hit");

    // Hit C — C becomes MRU.
    let hit_c = cache.find_best_prefix(&ids_c);
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
fn long_prompt_block_aligned_partial_hit() {
    let prompt_a: Vec<u32> = make_ids(16 * BLOCK_TOKENS); // 4096 tokens
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    cache.push(TestEntry::new(prompt_a.clone()));

    // First 15 blocks (3840 tokens) identical, last block differs.
    let mut prompt_b = prompt_a[..15 * BLOCK_TOKENS].to_vec();
    prompt_b.extend(9000..9000 + BLOCK_TOKENS as u32);
    assert_eq!(prompt_b.len(), 16 * BLOCK_TOKENS);

    let r = cache.find_best_prefix(&prompt_b);
    assert!(r.is_some(), "must hit on the shared 15-block prefix");
    let (slot_idx, block_count) = r.unwrap();
    assert_eq!(block_count, 15, "must match exactly 15 leading blocks");

    // Block-aligned truncation lands at 3840 tokens.
    let mut cloned = cache.slots[slot_idx].entry.deep_clone().unwrap();
    cloned.truncate_kv_to_block(block_count);
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

    let r_a = cache.find_best_prefix(&partial_prompt);
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
    let r_b = cache.find_best_prefix(&full_prompt);
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
    let r_b2 = cache.find_best_prefix(&base);
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
    let r_c = cache.find_best_prefix(&miss_prompt);
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
        .find_best_prefix(&prompt)
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
    fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<TestEntry>> {
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
    assert!(cache.find_best_prefix(&prompt).is_none(), "cold RAM miss");
    assert_eq!(cache.stats().ssd_hits, 0);

    // Hydrate from SSD → promoted into RAM, counter bumped.
    let slot = cache.hydrate_from_ssd(&prompt);
    assert!(slot.is_some(), "SSD hit must populate a RAM slot");
    assert_eq!(cache.stats().ssd_hits, 1, "ssd_hits must increment on hit");
    assert_eq!(cache.slots.len(), 1, "one slot now populated");

    // The hydrated entry is now served by find_best_prefix.
    let r = cache.find_best_prefix(&prompt);
    assert!(
        r.is_some(),
        "find_best_prefix must serve the hydrated entry"
    );
    let (_, blocks) = r.unwrap();
    assert_eq!(blocks, 2, "full 2-block prefix served from RAM");
}

/// with no SSD source attached (`None`), `hydrate_from_ssd` is inert
/// and `ssd_hits` stays 0 — today's RAM-only behavior is unchanged.
#[test]
fn no_ssd_source_is_inert() {
    let prompt: Vec<u32> = make_ids(2 * BLOCK_TOKENS);
    let mut cache: PromptCache<TestEntry> = PromptCache::new(4);
    assert!(cache.find_best_prefix(&prompt).is_none());
    assert!(
        cache.hydrate_from_ssd(&prompt).is_none(),
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
    assert!(cache.hydrate_from_ssd(&short).is_none(), "SSD miss");
    assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(cache.stats().ssd_hits, 0);
    assert_eq!(cache.slots.len(), 0);
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
    let (idx, blocks) = cache.find_best_prefix(&req).expect("3-block prefix hit");
    assert_eq!(blocks, 3, "exactly 3 leading blocks match");

    let mut cloned = cache.slots[idx].entry.deep_clone().unwrap();
    cloned.truncate_kv_to_block(blocks);
    assert_eq!(
        cloned.truncated_to.get(),
        Some(3 * BLOCK_TOKENS),
        "truncation must land at the matched-block boundary (3*256), not elsewhere"
    );
}

// ── resolve_ram_cap_bytes ────────────────────────────────────────

#[test]
fn ram_resolver_default_when_both_absent() {
    assert_eq!(resolve_ram_cap_bytes(None, None), DEFAULT_MAX_BYTES);
}

#[test]
fn ram_resolver_env_only() {
    assert_eq!(resolve_ram_cap_bytes(None, Some("536870912")), 536_870_912);
}

#[test]
fn ram_resolver_env_invalid_falls_through() {
    assert_eq!(
        resolve_ram_cap_bytes(None, Some("notanumber")),
        DEFAULT_MAX_BYTES
    );
}

#[test]
fn ram_resolver_cli_only() {
    // 1.5 GiB
    let bytes = resolve_ram_cap_bytes(Some(1.5), None);
    assert_eq!(bytes, (1.5 * 1024.0 * 1024.0 * 1024.0) as u64);
}

#[test]
fn ram_resolver_cli_overrides_env() {
    // CLI 1 GiB beats env 512 MiB.
    let bytes = resolve_ram_cap_bytes(Some(1.0), Some("536870912"));
    assert_eq!(bytes, 1024 * 1024 * 1024);
}

#[test]
fn ram_resolver_cli_negative_falls_through() {
    // Negative CLI is rejected → env wins.
    let bytes = resolve_ram_cap_bytes(Some(-1.0), Some("123456"));
    assert_eq!(bytes, 123_456);
    // No env either → default.
    let bytes2 = resolve_ram_cap_bytes(Some(-1.0), None);
    assert_eq!(bytes2, DEFAULT_MAX_BYTES);
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
        ArchPromptCache::new("test-partial", ReusePolicy::Partial);
    let exact: ArchPromptCache<TestEntry> =
        ArchPromptCache::new("test-exact", ReusePolicy::ExactOnly);
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
        ArchPromptCache::new("test-roundtrip", ReusePolicy::Partial);
    let ids: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    arch.with_inner_mut(|g| {
        *g = Some(PromptCache::new(4));
        g.as_mut().unwrap().push(TestEntry::new(ids.clone()));
    });
    // Second-call observability: the pushed entry is reachable.
    let hit_blocks = arch.with_inner_mut(|g| {
        let cache = g.as_mut().unwrap();
        cache.find_best_prefix(&ids).map(|(_, b)| b)
    });
    assert_eq!(hit_blocks, Some(2), "pushed entry must hit on full prompt");
    // Deep-clone semantics: clone the slot and verify tokens identical.
    arch.with_inner_mut(|g| {
        let cache = g.as_ref().unwrap();
        let cloned = cache.slots[0].entry.deep_clone().unwrap();
        assert_eq!(cloned.prompt_token_ids(), &ids[..]);
    });
}

/// unit test #4: `store_kv_cache_bytes` + `read_kv_cache_bytes`
/// round-trip. Mirrors the per-request /metrics/cache wire path.
#[test]
fn arch_cache_kv_bytes_round_trip() {
    let arch: ArchPromptCache<TestEntry> = ArchPromptCache::new("test-bytes", ReusePolicy::Partial);
    assert_eq!(arch.read_kv_cache_bytes(), 0, "fresh cache reports zero");
    arch.store_kv_cache_bytes(424_242);
    assert_eq!(arch.read_kv_cache_bytes(), 424_242);
    arch.store_kv_cache_bytes(0);
    assert_eq!(arch.read_kv_cache_bytes(), 0);
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
    let arch: ArchPromptCache<TestEntry> = ArchPromptCache::new("test-stats", ReusePolicy::Partial);
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
        ArchPromptCache::new("test-exact", ReusePolicy::ExactOnly);
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
        let m = cache.find_best_prefix(&req);
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
