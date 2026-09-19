//! Oracle for the per-arch `impl SsdHydrate<Entry> for SsdHydrator` blocks.
//!
//! ## What these tests read
//!
//! The observable is the state of the entry the arch impl returns, at the
//! deepest seam the change can reach:
//!
//! - the **per-layer bytes** of the restored `KvCache` vector, read back
//!   through `probe_k_dequant` / `probe_v_dequant` and folded into one
//!   FNV-1a-64 digest per layer;
//! - the **token vector** (`prompt_token_ids`), which is the block-aligned
//!   prefix the tier stored and not the request's full prompt;
//! - the **linear-attention state** for the one hybrid entry, read back
//!   through `LinearAttnCache::delta_state` / `conv_state`;
//! - the **cross-layer-KV topology** each restored cache carries
//!   (`KvCache::shares_kv`), which differs by arch and is the one argument the
//!   eight bodies do not agree on;
//! - the **hit/miss branch**: `Ok(Some(_))` against `Ok(None)`.
//!
//! No test here asserts that a function was called, and none reads a
//! resolver's return value. A hydrate that returns the right entry shape with
//! the wrong bytes in it is the defect these exist to catch.
//!
//! ## Why the digests are round-trip, not literal constants
//!
//! The spilled cache is digested before the write and the hydrated cache after
//! the read, and the two lists are compared. A literal constant would pin the
//! codec's packing as well as the hydrate wiring, so a legitimate codec change
//! would turn these red for a reason that has nothing to do with the entry
//! impls. The positive control that keeps a round-trip honest is
//! `assert_pairwise_distinct`: every fixture asserts its layers are
//! distinguishable before it asserts they came back in order, so "drops a
//! layer" and "reorders the layers" cannot both pass by accident.
//!
//! ## Why this runs on the CPU
//!
//! The whole spill/hydrate chain is device-parameterised and drives
//! `Device::Cpu` end to end: `write_caches` serialises the packed store,
//! `SsdHydrator::lookup_seeded` reads it back, and the dequant probes run on
//! the same device. No Metal context is taken, so none of these carries
//! `#[ignore]`.
//!
//! The fixtures write under a unique namespace below
//! `rmlx_core::paths::kv_cache_dir`, which is where the production hydrator
//! resolves its `.kvb` directory and index from. `SsdHydrator::with_index` —
//! the hermetic constructor the `rmlx-kv-ssd` unit tests use — is
//! `#[cfg(test)]` of that crate and cannot be reached from here, and
//! `rmlx_core::paths::home()` caches its resolution in a `OnceLock`, so an
//! environment override set from a test would be both process-global and
//! one-shot. A unique namespace is the hermetic unit that is actually
//! available. `Namespace` removes its directory on drop.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device};

use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_kv_ssd::{
    cache_seed, chained_block_hashes_seeded, hash_to_hex, write_caches, SsdHydrate, SsdHydrator,
    SsdKvIndex, BLOCK_TOKENS,
};

use crate::bitnet::prompt_cache::BitNetEntry;
use crate::gemma4::prompt_cache::Gemma4Entry;
use crate::laguna::prompt_cache::LagunaEntry;
use crate::prompt_cache::PromptCacheEntry;
use crate::qwen3::Qwen3Entry;
use crate::qwen3_5_moe::prompt_cache::Qwen35MoeEntry;

/// The codec every fixture spills under. `K8V8` keeps a packed store that
/// survives the round trip byte for byte, which is what makes a per-layer
/// digest comparison meaningful.
const QUANT: KvQuant = KvQuant::K8V8;

/// Non-zero on purpose. A zero model signature would let a probe that dropped
/// the model term still find these fixtures, and every test in the file would
/// keep passing against a hydrator that cannot find what a real model spilled.
const MODEL_SIG: u64 = 0x00c0_ffee_0bad_f00d;

/// Non-zero on purpose: it is the layout component of the digest seed, and
/// `a_block_does_not_hydrate_under_a_different_layout_key` needs a value a
/// mismatch can differ from.
const LAYOUT_KEY: u64 = 0x5eed_1a70_07de_c0de;

/// Head dimension of every fixture cache. A power of two, kept small so the
/// debug-profile probes stay cheap — the oracle reads layer identity and
/// ordering, neither of which depends on the width.
const HEAD_DIM: i32 = 32;

// ── namespace ───────────────────────────────────────────────────────────────

/// A `.kvb` directory plus index DB under `paths::kv_cache_dir`, unique to one
/// test and removed when the test ends.
struct Namespace {
    name: String,
    dir: PathBuf,
}

impl Namespace {
    fn new(tag: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let name = format!(
            "ssd-hydrate-oracle/{tag}-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        let dir = rmlx_core::paths::kv_cache_dir(&name);
        Self { name, dir }
    }
}

impl Drop for Namespace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// ── fixture data ────────────────────────────────────────────────────────────

/// Deterministic f32 stream in `[-1, 1]`. The same LCG the SSD block-I/O and
/// hydrate fixtures use, so a fixture moved between the two files keeps its
/// data.
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
    clippy::expect_used,
    reason = "test fixture: an Array that cannot be built from a known-length f32 slice is a fixture bug and must abort the test loudly"
)]
fn arr(data: &[f32], shape: &[i32]) -> Array {
    Array::from_f32_slice(data, shape).expect("fixture array")
}

/// One populated `KvCache` of `seq` tokens over `kv_h` KV heads.
///
/// Deliberately not bracketed by `enter_prefill` / `exit_prefill`: a prefilled
/// `K8V8` cache decodes off its bf16 mirror and `exit_prefill` builds no packed
/// store, so there would be nothing for `write_caches` to spill as a block. The
/// unbracketed append drives the codec body directly, which is the state a
/// hydrated cache is in.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a KV update that fails on a freshly built cache is a fixture bug and must abort the test loudly"
)]
fn kv_layer(seq: i32, kv_h: i32, seed: u64) -> KvCache {
    let mut c = KvCache::with_quant_max_seq(QUANT, 2 * BLOCK_TOKENS as i32);
    let shape = [1_i32, kv_h, seq, HEAD_DIM];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    c.update(
        &arr(&lcg(n, seed), &shape),
        &arr(&lcg(n, seed ^ 0xABCD), &shape),
        Device::Cpu,
    )
    .expect("fixture kv update");
    c
}

/// One populated GatedDeltaNet recurrent state.
fn lin_layer(seed: u64) -> LinearAttnCache {
    let conv_shape = [1_i32, 3, 16];
    let delta_shape = [1_i32, 2, 8, 8];
    let conv_n: usize = conv_shape.iter().map(|&x| x as usize).product();
    let delta_n: usize = delta_shape.iter().map(|&x| x as usize).product();
    LinearAttnCache {
        conv_state: Some(arr(&lcg(conv_n, seed), &conv_shape)),
        delta_state: Some(arr(&lcg(delta_n, seed ^ 0x1234), &delta_shape)),
        tape: None,
    }
}

// ── digests ─────────────────────────────────────────────────────────────────

fn fold(h: &mut u64, word: u64) {
    *h ^= word;
    *h = h.wrapping_mul(rmlx_kv_ssd::FNV_PRIME);
}

/// FNV-1a-64 over one cache's K dequant followed by its V dequant, plus the
/// restored sequence length. Two caches with the same digest hold the same
/// bytes at the same length.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a K8V8 cache with no probeable store, or a dequant that refuses, is a fixture bug and must abort the test loudly"
)]
fn digest_cache(c: &KvCache, device: Device) -> u64 {
    c.eval_gpu_state().expect("eval cache state");
    let mut h = rmlx_kv_ssd::FNV_OFFSET;
    fold(&mut h, c.seq_len() as u64);
    for probe in [
        c.probe_k_dequant(device)
            .expect("K probe: cache holds no q8 K store")
            .expect("K probe: dequant refused"),
        c.probe_v_dequant(device)
            .expect("V probe: cache holds no q8 V store")
            .expect("V probe: dequant refused"),
    ] {
        for x in probe {
            fold(&mut h, u64::from(x.to_bits()));
        }
    }
    h
}

fn digest_layers(caches: &[KvCache], device: Device) -> Vec<u64> {
    caches.iter().map(|c| digest_cache(c, device)).collect()
}

/// FNV-1a-64 over the raw bytes of one recurrent state's conv and delta
/// tensors.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a lin cache built with both tensors present that reads back without them is a fixture bug and must abort the test loudly"
)]
fn digest_lin(l: &LinearAttnCache) -> u64 {
    let mut h = rmlx_kv_ssd::FNV_OFFSET;
    for slot in [
        l.conv_state.as_ref().expect("lin conv_state present"),
        l.delta_state.as_ref().expect("lin delta_state present"),
    ] {
        slot.eval().expect("lin tensor eval");
        for b in slot.to_bytes().expect("lin tensor readback") {
            fold(&mut h, u64::from(b));
        }
    }
    h
}

/// Every fixture asserts this before it asserts an ordered round trip. Without
/// it a hydrate that returned layer 0 three times would pass the round-trip
/// comparison against a fixture whose layers happened to be equal.
fn assert_pairwise_distinct(digests: &[u64]) {
    for (i, a) in digests.iter().enumerate() {
        for (j, b) in digests.iter().enumerate().skip(i + 1) {
            assert_ne!(
                a, b,
                "fixture layers {i} and {j} are indistinguishable — an ordering \
                 assertion over them proves nothing"
            );
        }
    }
}

// ── spill ───────────────────────────────────────────────────────────────────

/// What a test needs to probe a block it has just written to the tier.
struct Spilled {
    /// The token ids the block covers — always a whole number of blocks.
    prompt_ids: Vec<u32>,
    /// The digest seed the probe has to run under.
    seed: u64,
    /// Per-layer digests of the caches as they went in.
    kv_digests: Vec<u64>,
    /// Per-layer digests of the recurrent state as it went in.
    lin_digests: Vec<u64>,
}

/// Write one block-aligned `.kvb` for `prompt_ids` and record its index row.
///
/// `layer_quants` is the per-layer codec vector the seed folds. It is passed
/// rather than derived so a test can spill under one mixture and probe under
/// another.
#[allow(
    clippy::expect_used,
    reason = "test fixture: index open / block write / file metadata failures are fixture bugs and must abort the test loudly"
)]
fn spill(
    ns: &Namespace,
    layout_key: u64,
    prompt_ids: &[u32],
    kv: &[KvCache],
    lin: &[LinearAttnCache],
) -> Spilled {
    let device = Device::Cpu;
    let layer_quants = vec![QUANT; kv.len()];
    let seed = cache_seed(layout_key, QUANT, &layer_quants, MODEL_SIG);

    let kv_digests = digest_layers(kv, device);
    let lin_digests = lin.iter().map(digest_lin).collect();

    let chained = chained_block_hashes_seeded(prompt_ids, seed);
    let key = hash_to_hex(*chained.last().expect("prompt covers a whole block"));
    let path = ns.dir.join(format!("{key}.kvb"));
    write_caches(&path, device, &ns.name, QUANT, kv, lin).expect("spill block");
    let size = std::fs::metadata(&path).expect("block metadata").len();

    let index = SsdKvIndex::open(&ns.name).expect("index open");
    index
        .record(&key, layout_key, &path, &ns.name, &QUANT.to_string(), size)
        .expect("index record");

    Spilled {
        prompt_ids: prompt_ids.to_vec(),
        seed,
        kv_digests,
        lin_digests,
    }
}

fn hydrator(ns: &Namespace, layout_key: u64) -> SsdHydrator {
    #[allow(
        clippy::expect_used,
        reason = "test fixture: a hydrator that cannot open the index the fixture just wrote is a fixture bug and must abort the test loudly"
    )]
    SsdHydrator::open(&ns.name, layout_key, Device::Cpu).expect("hydrator open")
}

fn ids(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

// ── pure attention, kv_h == 1 ───────────────────────────────────────────────

/// A `kv_h == 1` pure-attention entry gets back every layer it spilled, in
/// order, with the same bytes, and the block's tokens.
///
/// `kv_h == 1` is the shared-KV-head shape (gemma4-e2b and friends); the
/// layer-major byte layout collapses differently there than at `kv_h > 1`, so
/// a hydrate that mixes up its per-layer slicing can hold at one and fail at
/// the other.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a hydrate that misses where the fixture just spilled is the failure under test and must abort loudly"
)]
fn laguna_entry_restores_every_layer_in_order_at_kv_h_1() {
    let ns = Namespace::new("laguna-kv-h-1");
    let kv = vec![
        kv_layer(BLOCK_TOKENS as i32, 1, 0xA1),
        kv_layer(BLOCK_TOKENS as i32, 1, 0xB2),
        kv_layer(BLOCK_TOKENS as i32, 1, 0xC3),
    ];
    let prompt = ids(BLOCK_TOKENS);
    let s = spill(&ns, LAYOUT_KEY, &prompt, &kv, &[]);
    assert_pairwise_distinct(&s.kv_digests);

    let h = hydrator(&ns, LAYOUT_KEY);
    let entry: LagunaEntry = h
        .hydrate(&prompt, s.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error")
        .expect("the block the fixture spilled must be found");

    assert_eq!(
        digest_layers(entry.kv_caches(), Device::Cpu),
        s.kv_digests,
        "every layer must come back, in the order it was spilled, byte for byte"
    );
    assert_eq!(
        entry.prompt_token_ids(),
        s.prompt_ids.as_slice(),
        "the entry's tokens are the block's tokens"
    );
    assert_eq!(
        entry.kv_quant(),
        Some(QUANT),
        "the entry records the codec the request ran"
    );
    assert!(
        entry.is_ssd_hydrated(),
        "a hydrated entry must be flagged so the exact fast path excludes it — \
         replaying its placeholder first token poisons generation"
    );
}

// ── pure attention, kv_h > 1 ────────────────────────────────────────────────

/// The same oracle at `kv_h > 1`, on the entry that carries an extra field
/// beyond the shared seven.
///
/// `Qwen3Entry` sets `first_logprobs: None` on top of what the other
/// pure-attention entries set. A blanket impl has to keep that field's hydrate
/// value, which is what makes this entry the one worth reading at the second
/// shape.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a hydrate that misses where the fixture just spilled is the failure under test and must abort loudly"
)]
fn qwen3_entry_restores_every_layer_in_order_at_kv_h_4() {
    let ns = Namespace::new("qwen3-kv-h-4");
    let kv = vec![
        kv_layer(BLOCK_TOKENS as i32, 4, 0xD4),
        kv_layer(BLOCK_TOKENS as i32, 4, 0xE5),
        kv_layer(BLOCK_TOKENS as i32, 4, 0xF6),
        kv_layer(BLOCK_TOKENS as i32, 4, 0x17),
    ];
    let prompt = ids(BLOCK_TOKENS);
    let s = spill(&ns, LAYOUT_KEY, &prompt, &kv, &[]);
    assert_pairwise_distinct(&s.kv_digests);

    let h = hydrator(&ns, LAYOUT_KEY);
    let entry: Qwen3Entry = h
        .hydrate(&prompt, s.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error")
        .expect("the block the fixture spilled must be found");

    assert_eq!(
        digest_layers(entry.kv_caches(), Device::Cpu),
        s.kv_digests,
        "every layer must come back, in the order it was spilled, byte for byte"
    );
    assert_eq!(
        entry.prompt_token_ids(),
        s.prompt_ids.as_slice(),
        "the entry's tokens are the block's tokens"
    );
    assert!(entry.is_ssd_hydrated());
}

// ── hybrid ──────────────────────────────────────────────────────────────────

/// The hybrid entry restores its recurrent state beside its KV.
///
/// This is the one entry a pure-attention blanket impl must not swallow: the
/// seven others discard `lin_caches` from the block, and an eighth that did the
/// same would decode a Qwen3.5-MoE request from a zeroed GatedDeltaNet state.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a hydrate that misses where the fixture just spilled is the failure under test and must abort loudly"
)]
fn qwen3_5_moe_entry_restores_the_linear_state_beside_the_kv() {
    let ns = Namespace::new("qwen35moe-hybrid");
    let kv = vec![
        kv_layer(BLOCK_TOKENS as i32, 2, 0x28),
        kv_layer(BLOCK_TOKENS as i32, 2, 0x39),
    ];
    let lin = vec![lin_layer(0x4A), lin_layer(0x5B)];
    let prompt = ids(BLOCK_TOKENS);
    let s = spill(&ns, LAYOUT_KEY, &prompt, &kv, &lin);
    assert_pairwise_distinct(&s.kv_digests);
    assert_pairwise_distinct(&s.lin_digests);

    let h = hydrator(&ns, LAYOUT_KEY);
    let entry: Qwen35MoeEntry = h
        .hydrate(&prompt, s.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error")
        .expect("the block the fixture spilled must be found");

    assert_eq!(
        digest_layers(entry.kv_caches(), Device::Cpu),
        s.kv_digests,
        "every KV layer must come back, in order, byte for byte"
    );
    assert_eq!(
        entry
            .lin_caches()
            .iter()
            .map(digest_lin)
            .collect::<Vec<_>>(),
        s.lin_digests,
        "the recurrent state must come back too — a pure-attention hydrate \
         would leave this empty and decode from a zeroed GDN state"
    );
    assert_eq!(entry.prompt_token_ids(), s.prompt_ids.as_slice());
    assert!(entry.is_ssd_hydrated());
}

// ── the argument the eight bodies do not agree on ───────────────────────────

/// Each arch's hydrate carries that arch's own cross-layer-KV topology.
///
/// Seven of the eight bodies pass a literal `false`; gemma4 passes its own
/// `SHARES_KV_ACROSS_LAYERS`, which is the only `true` in the tree. The flag
/// lands on every restored cache, and a hydrated cache can be tail-extended,
/// which re-runs the `exit_prefill` gate this flag decides. A shared body that
/// hard-codes either value is right for one arch and silently wrong for the
/// other, so both are read in one test.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a hydrate that misses where the fixture just spilled is the failure under test and must abort loudly"
)]
fn each_arch_hydrates_under_its_own_cross_layer_kv_topology() {
    // The pair below has to span both values of the flag. A tree where both
    // archs agree makes every assertion in this test satisfiable by one
    // hard-coded literal, so it stops compiling instead.
    const {
        assert!(crate::gemma4::SHARES_KV_ACROSS_LAYERS);
        assert!(!crate::laguna::SHARES_KV_ACROSS_LAYERS);
    }

    let prompt = ids(BLOCK_TOKENS);

    let gemma_ns = Namespace::new("gemma4-shares-kv");
    let g = spill(
        &gemma_ns,
        LAYOUT_KEY,
        &prompt,
        &[kv_layer(BLOCK_TOKENS as i32, 2, 0x6C)],
        &[],
    );
    let gemma: Gemma4Entry = hydrator(&gemma_ns, LAYOUT_KEY)
        .hydrate(&prompt, g.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error")
        .expect("the block the fixture spilled must be found");
    assert!(
        gemma.kv_caches().iter().all(KvCache::shares_kv),
        "a gemma4 hydrate must build caches at gemma4's topology — a hard-coded \
         `false` here drops a bf16 mirror the arch reads"
    );

    let laguna_ns = Namespace::new("laguna-no-shares-kv");
    let l = spill(
        &laguna_ns,
        LAYOUT_KEY,
        &prompt,
        &[kv_layer(BLOCK_TOKENS as i32, 2, 0x7D)],
        &[],
    );
    let laguna: LagunaEntry = hydrator(&laguna_ns, LAYOUT_KEY)
        .hydrate(&prompt, l.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error")
        .expect("the block the fixture spilled must be found");
    assert!(
        !laguna.kv_caches().iter().any(KvCache::shares_kv),
        "a laguna hydrate must build caches at laguna's topology — a hard-coded \
         `true` here builds a mirror the arch never reads"
    );
}

// ── the tokens come from the block, not the request ─────────────────────────

/// A request longer than the stored block gets the block's tokens back, not
/// its own.
///
/// The entry's `prompt_token_ids` is what the RAM cache later compares a repeat
/// against. An impl that echoed the request's ids would claim the tail tokens
/// are prefilled when they are not, and the generate loop would skip the
/// re-prefill that computes them.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: a hydrate that misses where the fixture just spilled is the failure under test and must abort loudly"
)]
fn the_entry_carries_the_blocks_tokens_not_the_requests() {
    let ns = Namespace::new("block-aligned-tokens");
    let stored = ids(BLOCK_TOKENS);
    let s = spill(
        &ns,
        LAYOUT_KEY,
        &stored,
        &[kv_layer(BLOCK_TOKENS as i32, 2, 0x8E)],
        &[],
    );

    let mut request = stored.clone();
    request.extend(BLOCK_TOKENS as u32..BLOCK_TOKENS as u32 + 40);

    let entry: BitNetEntry = hydrator(&ns, LAYOUT_KEY)
        .hydrate(&request, s.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error")
        .expect("the longer request shares the stored block as its prefix");

    assert_eq!(
        entry.prompt_token_ids(),
        stored.as_slice(),
        "the entry holds the block-aligned prefix that was actually prefilled"
    );
    assert_eq!(
        entry
            .kv_caches()
            .first()
            .expect("a hydrated entry holds the layer the fixture spilled")
            .seq_len(),
        BLOCK_TOKENS as i32,
        "and its KV is that prefix's length, not the request's"
    );
}

// ── the miss branch ─────────────────────────────────────────────────────────

/// A prompt the tier never saw is `Ok(None)`, not an error and not a stale
/// entry.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: hydrate is documented never to return Err; a failure here is the behaviour under test"
)]
fn a_prompt_the_tier_never_saw_is_a_miss() {
    let ns = Namespace::new("unknown-prompt");
    let stored = ids(BLOCK_TOKENS);
    let s = spill(
        &ns,
        LAYOUT_KEY,
        &stored,
        &[kv_layer(BLOCK_TOKENS as i32, 2, 0x9F)],
        &[],
    );

    let other: Vec<u32> = (9_000..9_000 + BLOCK_TOKENS as u32).collect();
    let miss: Option<LagunaEntry> = hydrator(&ns, LAYOUT_KEY)
        .hydrate(&other, s.seed, QUANT, DispatchPolicy::default())
        .expect("a miss is Ok(None), never Err");
    assert!(
        miss.is_none(),
        "a prompt with no indexed prefix must miss, so the caller re-prefills"
    );
}

/// The layout key is part of what a probe matches on.
///
/// The key is the SSD tier's shape identity. A block written under one layout
/// must not hydrate into a request running another — the bytes would be read
/// back under a geometry they were not packed at.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test: hydrate is documented never to return Err; a failure here is the behaviour under test"
)]
fn a_block_does_not_hydrate_under_a_different_layout_key() {
    let ns = Namespace::new("layout-key");
    let prompt = ids(BLOCK_TOKENS);
    let s = spill(
        &ns,
        LAYOUT_KEY,
        &prompt,
        &[kv_layer(BLOCK_TOKENS as i32, 2, 0xAB)],
        &[],
    );

    let hit: Option<LagunaEntry> = hydrator(&ns, LAYOUT_KEY)
        .hydrate(&prompt, s.seed, QUANT, DispatchPolicy::default())
        .expect("hydrate must not error");
    assert!(
        hit.is_some(),
        "positive control: the matching layout key must hit, or the negative \
         below proves nothing"
    );

    let other_key = LAYOUT_KEY ^ 1;
    let miss: Option<LagunaEntry> = hydrator(&ns, other_key)
        .hydrate(
            &prompt,
            cache_seed(other_key, QUANT, &[QUANT], MODEL_SIG),
            QUANT,
            DispatchPolicy::default(),
        )
        .expect("hydrate must not error");
    assert!(
        miss.is_none(),
        "a block stored under one layout key must not be served to a request \
         running another"
    );
}
