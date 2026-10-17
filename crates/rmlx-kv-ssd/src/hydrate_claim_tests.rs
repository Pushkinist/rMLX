//! A hydrated block sits at the prefix its `prompt_ids` claim.
//!
//! Child of `hydrate_tests` so it can reuse that module's fixtures through
//! `super::*`. Each case spills an entry of a given length through the real
//! block writer, records it under the digest of its last full block (as the
//! spill sink does) and hydrates it for a longer request over the same tokens.

#![allow(
    clippy::indexing_slicing,
    reason = "test: slices and layer indexing over fixtures whose sizes the test fixed"
)]

use super::*;
use crate::block_io::{read_caches_timed, write_caches_timed, BlockRead, ClaimMismatch};

const BLOCK: usize = BLOCK_TOKENS;
const HEADS: usize = 2;
const HEAD_DIM: usize = 128;

/// Which state a spilled cache is in.
#[derive(Clone, Copy)]
enum Route {
    /// An unbracketed append: K/V lives in the codec's packed store.
    PackedStore,
    /// The production prefill route: K/V lives in the bf16 mirror and the store
    /// is geometry-only.
    Bf16Mirror,
}

/// A namespace holding the blocks a test spills, plus a hydrator over it.
struct Tier {
    hydrator: SsdHydrator,
    quant: KvQuant,
    seed: u64,
    dir: TempDir,
}

impl Tier {
    #[allow(
        clippy::unwrap_used,
        reason = "test: index open on a temp dir this test owns cannot fail"
    )]
    fn new() -> Self {
        Self::with_quant(QUANT)
    }

    #[allow(
        clippy::unwrap_used,
        reason = "test: index open on a temp dir this test owns cannot fail"
    )]
    fn with_quant(quant: KvQuant) -> Self {
        let device = Device::Cpu;
        let dir = TempDir::new().unwrap();
        let index = SsdKvIndex::open_at(&dir.path().join("index.db")).unwrap();
        Self {
            hydrator: SsdHydrator::with_index(
                MODEL_ID,
                TEST_LAYOUT_KEY,
                device,
                dir.path().to_path_buf(),
                index,
            ),
            quant,
            seed: cache_seed(TEST_LAYOUT_KEY, quant, &[quant], TEST_MODEL_SIG),
            dir,
        }
    }

    fn ids(len: usize) -> Vec<u32> {
        (0..len as u32).collect()
    }

    /// The key a block of `key_len` tokens is recorded under: the digest of its
    /// last full block.
    #[allow(
        clippy::expect_used,
        reason = "test: a key length under one block is a fixture bug"
    )]
    fn key(&self, key_len: usize) -> String {
        let chained = chained_block_hashes_seeded(&Self::ids(key_len), self.seed);
        hash_to_hex_local(*chained.last().expect("key covers a full block"))
    }

    fn path(&self, key_len: usize) -> PathBuf {
        self.dir.path().join(format!("{}.kvb", self.key(key_len)))
    }

    /// Spill `cache` (and `lin`) as the entry for a prompt of `key_len` tokens.
    #[allow(
        clippy::expect_used,
        reason = "test: a fixture that fails to spill or record is a fixture bug and must abort loudly"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "test: file metadata on a temp dir this test owns cannot fail"
    )]
    fn spill(&self, key_len: usize, cache: &KvCache, lin: &[LinearAttnCache]) {
        let path = self.path(key_len);
        let state_len = cache.offset();
        write_caches_timed(
            &path,
            Device::Cpu,
            MODEL_ID,
            self.quant,
            std::slice::from_ref(cache),
            lin,
            state_len,
        )
        .expect("spill");
        let size = std::fs::metadata(&path).unwrap().len();
        self.hydrator
            .index
            .record(
                &self.key(key_len),
                TEST_LAYOUT_KEY,
                &path,
                MODEL_ID,
                &self.quant.to_string(),
                size,
            )
            .expect("record");
    }

    /// Hydrate for a request of `request_len` tokens over the spilled prompts.
    #[allow(
        clippy::expect_used,
        reason = "test: a caller error from a one-layer fixture is a fixture bug"
    )]
    fn lookup(&self, request_len: usize) -> Option<HydratedBlock> {
        self.hydrator
            .lookup(
                &Self::ids(request_len),
                self.seed,
                self.quant,
                &[self.quant],
                DispatchPolicy::default(),
                false,
            )
            .expect("lookup must not error")
    }

    #[allow(
        clippy::unwrap_used,
        reason = "test: an index read on a temp db this test owns cannot fail"
    )]
    fn has_row(&self, key_len: usize) -> bool {
        self.hydrator
            .index
            .lookup(&self.key(key_len), TEST_LAYOUT_KEY)
            .unwrap()
            .is_some()
    }
}

#[allow(
    clippy::unwrap_used,
    reason = "test: a CPU prefill of a correctly shaped cache cannot fail"
)]
fn build_cache(route: Route, len: i32, seed: u64) -> KvCache {
    match route {
        Route::PackedStore => build_kvcache(len, seed),
        Route::Bf16Mirror => {
            let shape = [1i32, HEADS as i32, len, HEAD_DIM as i32];
            let n: usize = shape.iter().map(|&x| x as usize).product();
            let mut cache = KvCache::with_quant_max_seq(QUANT, 4096);
            cache.enter_prefill();
            cache
                .update(
                    &arr(&lcg(n, seed), &shape),
                    &arr(&lcg(n, seed ^ 0xABCD), &shape),
                    Device::Cpu,
                )
                .unwrap();
            cache.exit_prefill(Device::Cpu).unwrap();
            cache
        }
    }
}

fn recurrent_state() -> Vec<LinearAttnCache> {
    let mut state = LinearAttnCache::new();
    state.conv_state = Some(arr(&lcg(24, 1), &[1, 3, 8]));
    state.delta_state = Some(arr(&lcg(48, 2), &[1, 2, 4, 6]));
    vec![state]
}

#[allow(
    clippy::expect_used,
    reason = "test: a V store of a K8V8 cache with a refusing dequant is a bug under test"
)]
fn probe_v(cache: &KvCache) -> Vec<f32> {
    cache
        .probe_v_dequant(Device::Cpu)
        .expect("cache holds a V store")
        .expect("V dequant must not refuse")
}

/// `cut` holds the first `cut_seq` positions of every head of `full`.
fn assert_is_prefix(full: &[f32], full_seq: usize, cut: &[f32], cut_seq: usize) {
    assert_eq!(cut.len(), HEADS * cut_seq * HEAD_DIM, "cut length");
    for head in 0..HEADS {
        let from = head * full_seq * HEAD_DIM;
        let kept = &full[from..from + cut_seq * HEAD_DIM];
        let got = &cut[head * cut_seq * HEAD_DIM..(head + 1) * cut_seq * HEAD_DIM];
        assert_eq!(
            kept, got,
            "head {head} must keep its first {cut_seq} positions"
        );
    }
}

/// The bf16 mirror of `cache` as raw bytes, K then V.
#[allow(
    clippy::expect_used,
    reason = "test: a mirror-route cache without a mirror is a fixture bug"
)]
fn seed_bytes(cache: &KvCache) -> (Vec<u8>, Vec<u8>, i32) {
    let (k, v) = cache.decode_fp16_kv().expect("cache holds a bf16 mirror");
    k.eval().expect("eval");
    v.eval().expect("eval");
    (
        k.to_bytes().expect("k bytes"),
        v.to_bytes().expect("v bytes"),
        k.shape()[2],
    )
}

/// `cut` holds the first `cut_seq` positions of every head of `full`.
fn assert_bytes_are_prefix(full: &[u8], full_seq: usize, cut: &[u8], cut_seq: usize) {
    let per_pos = full.len() / (HEADS * full_seq);
    assert_eq!(cut.len(), HEADS * cut_seq * per_pos, "cut byte length");
    for head in 0..HEADS {
        let from = head * full_seq * per_pos;
        assert_eq!(
            &full[from..from + cut_seq * per_pos],
            &cut[head * cut_seq * per_pos..(head + 1) * cut_seq * per_pos],
            "head {head} seed must keep its first {cut_seq} positions"
        );
    }
}

/// K/V past the last full block is cut back to it, so the caches sit where the
/// ids say they do and hold the same values at the kept positions.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a lookup that must hit cannot be absent"
)]
fn kv_past_the_claim_is_cut_back_to_it() {
    let tier = Tier::new();
    let len = BLOCK + 44;
    let cache = build_cache(Route::PackedStore, len as i32, 0x51D1);
    let (k_before, v_before) = (probe_k(&cache, Device::Cpu), probe_v(&cache));
    tier.spill(len, &cache, &[]);

    let block = tier.lookup(len + 40).expect("pure K/V block must hydrate");

    assert_eq!(
        block.prompt_ids.len(),
        BLOCK,
        "label is the last full block"
    );
    assert_eq!(block.kv_caches[0].offset(), BLOCK as i32);
    assert_is_prefix(
        &k_before,
        len,
        &probe_k(&block.kv_caches[0], Device::Cpu),
        BLOCK,
    );
    assert_is_prefix(&v_before, len, &probe_v(&block.kv_caches[0]), BLOCK);
}

/// On the production prefill route the bf16 mirror IS the K/V. It is cut with
/// the offset, so the hydrated cache can be spilled again and hydrated again
/// with the same contents.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a lookup that must hit, and a re-spill that must be accepted, cannot be absent"
)]
fn a_bf16_mirror_block_is_cut_with_its_seed_and_survives_a_respill() {
    let tier = Tier::new();
    let len = BLOCK + 44;
    let cache = build_cache(Route::Bf16Mirror, len as i32, 0x2468);
    let (k_before, v_before, seed_len) = seed_bytes(&cache);
    assert_eq!(seed_len, len as i32, "fixture: the mirror holds the prompt");
    tier.spill(len, &cache, &[]);

    let first = tier.lookup(len + 40).expect("mirror block must hydrate");
    let cut = &first.kv_caches[0];
    let (k_cut, v_cut, cut_len) = seed_bytes(cut);
    assert_eq!(cut.offset(), BLOCK as i32);
    assert_eq!(cut_len, BLOCK as i32, "the mirror is cut with the offset");
    assert_bytes_are_prefix(&k_before, len, &k_cut, BLOCK);
    assert_bytes_are_prefix(&v_before, len, &v_cut, BLOCK);

    // The hydrated entry is evicted from RAM and spilled again under the key of
    // its last full block, which is the key the first spill used.
    let path = tier.path(BLOCK);
    write_caches_timed(
        &path,
        Device::Cpu,
        MODEL_ID,
        QUANT,
        &first.kv_caches,
        &first.lin_caches,
        BLOCK as i32,
    )
    .expect("a hydrated cache must be spillable again");

    let second = tier.lookup(len + 40).expect("block must hydrate again");
    let (k_again, v_again, again_len) = seed_bytes(&second.kv_caches[0]);
    assert_eq!(second.kv_caches[0].offset(), BLOCK as i32);
    assert_eq!((k_again, v_again, again_len), (k_cut, v_cut, cut_len));
}

/// A recurrent state that is not at the claim cannot be rewound: the block is a
/// miss and is deleted, so the next lookup reaches the shorter block it was
/// shadowing instead of reading the unusable one again.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a lookup that must hit cannot be absent"
)]
fn a_recurrent_state_past_the_claim_is_a_miss_and_the_shorter_block_serves_next() {
    let tier = Tier::new();
    let usable = BLOCK;
    let unusable = 2 * BLOCK + 44;
    tier.spill(
        usable,
        &build_cache(Route::PackedStore, usable as i32, 0x11),
        &recurrent_state(),
    );
    tier.spill(
        unusable,
        &build_cache(Route::PackedStore, unusable as i32, 0x22),
        &recurrent_state(),
    );
    let request = unusable + 40;

    assert!(
        tier.lookup(request).is_none(),
        "state is at the wrong token"
    );
    assert!(!tier.has_row(unusable), "the unusable row is deleted");
    assert!(!tier.path(unusable).exists(), "and its file");
    assert!(tier.has_row(usable), "the shorter block is untouched");

    let block = tier.lookup(request).expect("the shorter block now serves");
    assert_eq!(block.prompt_ids.len(), BLOCK);
    assert_eq!(block.kv_caches[0].offset(), BLOCK as i32);
    assert_eq!(block.lin_caches.len(), 1, "recurrent state is kept");
}

/// Control: a block whose length is a multiple of the block size already sits
/// at its claim, recurrent state included.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a lookup that must hit cannot be absent"
)]
fn a_recurrent_state_at_the_claim_resumes() {
    let tier = Tier::new();
    let len = 2 * BLOCK;
    tier.spill(
        len,
        &build_cache(Route::PackedStore, len as i32, 0x33),
        &recurrent_state(),
    );

    let block = tier.lookup(len + 40).expect("aligned block must hydrate");

    assert_eq!(block.prompt_ids.len(), len);
    assert_eq!(block.kv_caches[0].offset(), len as i32);
    assert_eq!(block.lin_caches.len(), 1, "recurrent state is kept");
}

/// A block that records no position for its recurrent state (an older binary's)
/// is not reused, whatever the K/V length says.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: the block this test just wrote records lin_pos"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test: file access on a temp dir this test owns cannot fail"
)]
fn a_recurrent_block_without_a_recorded_position_is_a_miss() {
    let tier = Tier::new();
    let len = BLOCK;
    tier.spill(
        len,
        &build_cache(Route::PackedStore, len as i32, 0x44),
        &recurrent_state(),
    );
    // Same-length rename keeps the safetensors header size valid.
    let path = tier.path(len);
    let mut bytes = std::fs::read(&path).unwrap();
    let at = bytes
        .windows(9)
        .position(|w| w == b"\"lin_pos\"")
        .expect("block records lin_pos");
    bytes[at..at + 9].copy_from_slice(b"\"lin_pox\"");
    std::fs::write(&path, bytes).unwrap();

    assert!(tier.lookup(len + 40).is_none());
    assert!(!tier.has_row(len), "the block is deleted");
}

/// A store refuses a cut at batch > 1 by leaving its CPU payload longer than its
/// shape, with no error from the cut itself. The block is a miss, not a cache
/// that decodes the discarded tokens.
#[test]
fn a_store_that_refuses_the_cut_is_a_miss() {
    let tier = Tier::with_quant(KvQuant::Iso3Sym);
    let len = BLOCK + 44;
    tier.spill(len, &batched_iso_cache(len), &[]);

    assert!(tier.lookup(len + 40).is_none());
    assert!(!tier.has_row(BLOCK), "the block is deleted");
}

/// The reader names why a block is unusable, and decides the three header
/// reasons before it reads a tensor.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test: a read of a block this test just wrote cannot fail"
)]
fn the_reader_names_why_a_block_is_unusable() {
    let why = |tier: &Tier, key_len: usize, claim: usize| match read_caches_timed(
        &tier.path(key_len),
        Device::Cpu,
        MODEL_ID,
        tier.quant,
        &[tier.quant],
        DispatchPolicy::default(),
        false,
        claim as i32,
    )
    .unwrap()
    {
        BlockRead::Unusable(why) => why,
        BlockRead::Gone => panic!("block must exist"),
        BlockRead::Caches(_) => panic!("block must be unusable"),
    };

    let tier = Tier::new();
    let len = 2 * BLOCK + 44;
    tier.spill(
        len,
        &build_cache(Route::PackedStore, len as i32, 0x77),
        &recurrent_state(),
    );
    assert_eq!(
        why(&tier, len, 2 * BLOCK),
        ClaimMismatch::StateNotAtClaim {
            state: len as i32,
            claim: 2 * BLOCK as i32
        }
    );
    assert_eq!(
        why(&tier, len, 3 * BLOCK),
        ClaimMismatch::KvShorter {
            held: len as i32,
            claim: 3 * BLOCK as i32
        }
    );

    let tier = Tier::with_quant(KvQuant::Iso3Sym);
    let len = BLOCK + 44;
    let cache = batched_iso_cache(len);
    tier.spill(len, &cache, &[]);
    assert_eq!(
        why(&tier, len, BLOCK),
        ClaimMismatch::CutRefused {
            layer: 0,
            claim: BLOCK as i32
        }
    );
}

/// A store that cannot be cut: a batch of 2 is accepted by the iso codec and
/// refused when a cut lands inside a block.
#[allow(
    clippy::unwrap_used,
    reason = "test: a CPU append of a correctly shaped batch cannot fail"
)]
fn batched_iso_cache(len: usize) -> KvCache {
    let shape = [2i32, HEADS as i32, len as i32, 96];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let mut cache = KvCache::with_quant_max_seq(KvQuant::Iso3Sym, 4096);
    cache
        .update(
            &arr(&lcg(n, 0x66), &shape),
            &arr(&lcg(n, 0x67), &shape),
            Device::Cpu,
        )
        .unwrap();
    cache
}

/// K/V shorter than the claim cannot be grown: the block is a miss.
#[test]
fn kv_shorter_than_the_claim_is_a_miss() {
    let tier = Tier::new();
    let held = BLOCK + 44;
    // Recorded under the key of a two-block prompt but holding 300 tokens.
    tier.spill(
        2 * BLOCK,
        &build_cache(Route::PackedStore, held as i32, 0x55),
        &[],
    );

    assert!(tier.lookup(2 * BLOCK + 40).is_none());
    assert!(!tier.has_row(2 * BLOCK), "the block is deleted");
}
