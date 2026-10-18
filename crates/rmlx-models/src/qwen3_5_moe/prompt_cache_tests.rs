use super::*;
use crate::prompt_cache::{
    anchor_target, chained_block_hashes_seeded, entry_is_storable, request_cache_seed, Consumed,
    PromptCache, BLOCK_TOKENS,
};
use rmlx_core::error::Error;
use rmlx_mlx::{Array, Device};

const TEST_QUANT: KvQuant = KvQuant::K8V8;
const TEST_LAYERS: usize = 4;
const TEST_SIG: u64 = 0x2222_3333_4444_5555;

/// Width of the entry's own recurrent state and of its anchor, so a test can
/// tell which of the two a reuse handed back.
const END_STATE_WIDTH: i32 = 3;
const ANCHOR_STATE_WIDTH: i32 = 5;

fn ids(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// One layer's recurrent state, `width` wide on the conv axis.
#[allow(
    clippy::unwrap_used,
    reason = "test-only: Array::from_f32_slice cannot fail on a correctly-sized F32 buffer"
)]
fn lin(width: i32) -> Vec<LinearAttnCache> {
    let mut cache = LinearAttnCache::new();
    cache.conv_state =
        Some(Array::from_f32_slice(&vec![0.0; width as usize], &[1, 1, width]).unwrap());
    vec![cache]
}

fn conv_width(lin_caches: &[LinearAttnCache]) -> Option<i32> {
    let shape = lin_caches.first()?.conv_state.as_ref()?.shape();
    shape.get(2).copied()
}

/// A full-attention cache holding `n` positions.
#[allow(
    clippy::unwrap_used,
    reason = "test-only: the CPU update path is infallible on a correctly-shaped fixture"
)]
fn kv_holding(n: i32) -> KvCache {
    let data: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
    let kv = Array::from_f32_slice(&data, &[1, 1, n, 2]).unwrap();
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 4096);
    cache.update(&kv, &kv, Device::Cpu).unwrap();
    cache
}

/// A RAM entry for a prompt of `ids`, its digests seeded as `consume` queries.
fn entry_of(ids: Vec<u32>, anchor_pos: Option<usize>, kv_caches: Vec<KvCache>) -> Qwen35MoeEntry {
    let seed = request_cache_seed(
        0,
        TEST_QUANT,
        TEST_LAYERS,
        crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS,
        TEST_SIG,
    );
    Qwen35MoeEntry {
        block_hashes: chained_block_hashes_seeded(&ids, seed),
        prompt_token_ids: ids,
        kv_caches,
        lin_caches: lin(END_STATE_WIDTH),
        anchor: anchor_pos.map(|pos| Anchor {
            pos,
            lin_caches: lin(ANCHOR_STATE_WIDTH),
        }),
        first_id: 7,
        first_piece: String::new(),
        first_logprobs: None,
        kv_quant: Some(TEST_QUANT),
        is_ssd_hydrated: false,
    }
}

/// An entry with the anchor a prefill of `len` tokens captures.
fn entry(len: usize) -> Qwen35MoeEntry {
    entry_of(ids(len), Anchor::pos_for(len), Vec::new())
}

/// The first `from` tokens of `base`, then `extra` tokens past its end that
/// `base` does not hold.
fn diverging_at(base: &[u32], from: usize, extra: usize) -> Vec<u32> {
    let tail = (0..(base.len() - from + extra) as u32).map(|i| 1_000_000 + i);
    base.iter().copied().take(from).chain(tail).collect()
}

/// The first `n` tokens of `base`.
fn head(base: &[u32], n: usize) -> Vec<u32> {
    base.iter().copied().take(n).collect()
}

fn reuse(entry: &Qwen35MoeEntry, prompt: &[u32]) -> Option<ReuseKind> {
    entry.is_reusable_prefix_of(prompt, false, prompt.len() / BLOCK_TOKENS)
}

fn anchor_resume(effective_blocks: usize) -> ReuseKind {
    ReuseKind::BlockTruncate { effective_blocks }
}

#[test]
fn anchor_sits_at_the_last_full_block_that_leaves_a_tail() {
    assert_eq!(Anchor::pos_for(0), None);
    assert_eq!(
        Anchor::pos_for(256),
        None,
        "a block-sized prompt has no tail past it"
    );
    assert_eq!(Anchor::pos_for(257), Some(256));
    assert_eq!(Anchor::pos_for(512), Some(256));
    assert_eq!(Anchor::pos_for(700), Some(512));
}

#[test]
fn a_strict_prefix_resumes_at_the_stored_length() {
    let stored = entry(700);
    let prompt = [stored.prompt_token_ids.clone(), ids(50)].concat();
    assert_eq!(
        reuse(&stored, &prompt),
        Some(ReuseKind::StrictPrefix { prefix_len: 700 })
    );
}

#[test]
fn an_equal_prompt_is_left_to_the_exact_arm() {
    let stored = entry(700);
    assert_eq!(reuse(&stored, &stored.prompt_token_ids.clone()), None);
}

#[test]
fn a_prompt_that_differs_in_the_stored_last_token_resumes_at_the_anchor() {
    let stored = entry(700);
    let prompt = diverging_at(&stored.prompt_token_ids, 699, 60);
    assert_eq!(reuse(&stored, &prompt), Some(anchor_resume(2)));
}

#[test]
fn the_anchor_is_offered_up_to_the_first_differing_token_and_no_further() {
    let stored = entry(700);
    let stored_ids = &stored.prompt_token_ids;
    assert_eq!(
        reuse(&stored, &diverging_at(stored_ids, 512, 60)),
        Some(anchor_resume(2)),
        "the anchor at 512 is at the first differing token"
    );
    assert_eq!(
        reuse(&stored, &diverging_at(stored_ids, 511, 60)),
        None,
        "the state at 512 has consumed a token the request does not hold"
    );
    assert_eq!(reuse(&stored, &diverging_at(stored_ids, 300, 60)), None);
}

#[test]
fn an_anchor_the_request_does_not_pass_is_not_offered() {
    let stored = entry(700);
    assert_eq!(
        reuse(&stored, &head(&stored.prompt_token_ids, 512)),
        None,
        "no tail would be left to forward"
    );
    assert_eq!(
        reuse(&stored, &head(&stored.prompt_token_ids, 513)),
        Some(anchor_resume(2))
    );
}

#[test]
fn an_entry_without_an_anchor_offers_only_the_strict_prefix() {
    let stored = entry_of(ids(700), None, Vec::new());
    let diverged = diverging_at(&stored.prompt_token_ids, 699, 60);
    assert_eq!(reuse(&stored, &diverged), None);
    let extended = [stored.prompt_token_ids.clone(), ids(5)].concat();
    assert_eq!(
        reuse(&stored, &extended),
        Some(ReuseKind::StrictPrefix { prefix_len: 700 })
    );
}

#[test]
fn an_entry_whose_caches_cannot_be_cut_to_the_anchor_does_not_offer_it() {
    // A cache holding 300 positions has nothing at 512 to cut back to.
    let short = entry_of(ids(700), Some(512), vec![kv_holding(300)]);
    let prompt = diverging_at(&short.prompt_token_ids, 699, 60);
    assert_eq!(reuse(&short, &prompt), None);

    let whole = entry_of(ids(700), Some(512), vec![kv_holding(700)]);
    assert_eq!(reuse(&whole, &prompt), Some(anchor_resume(2)));
}

#[test]
fn the_anchor_reuse_hands_back_the_anchor_state_and_the_cut_caches() -> Result<()> {
    let stored = entry_of(ids(700), Some(512), vec![kv_holding(700)]);
    let prepared = stored.prepare_reuse(ReuseKind::BlockTruncate {
        effective_blocks: 2,
    })?;
    assert_eq!(conv_width(&prepared.lin_caches), Some(ANCHOR_STATE_WIDTH));
    assert_eq!(
        prepared.kv_caches.first().map(KvCache::offset),
        Some(512),
        "the K/V is cut to the anchor"
    );
    assert_eq!(
        prepared.anchor.as_ref().map(|a| a.pos),
        Some(512),
        "the resumed entry still names the anchor it started from"
    );
    assert_eq!(
        conv_width(&stored.lin_caches),
        Some(END_STATE_WIDTH),
        "the stored entry keeps its own end state"
    );
    assert_eq!(
        stored.kv_caches.first().map(KvCache::offset),
        Some(700),
        "reuse leaves the stored caches alone"
    );
    Ok(())
}

#[test]
fn a_strict_prefix_reuse_keeps_the_end_state() -> Result<()> {
    let stored = entry(700);
    let prepared = stored.prepare_reuse(ReuseKind::StrictPrefix { prefix_len: 700 })?;
    assert_eq!(conv_width(&prepared.lin_caches), Some(END_STATE_WIDTH));
    Ok(())
}

#[test]
fn an_anchor_reuse_the_entry_has_no_state_for_is_refused() {
    let stored = entry_of(ids(700), Some(512), Vec::new());
    assert!(stored
        .prepare_reuse(ReuseKind::BlockTruncate {
            effective_blocks: 1
        })
        .is_err());
    let bare = entry_of(ids(700), None, Vec::new());
    assert!(bare
        .prepare_reuse(ReuseKind::BlockTruncate {
            effective_blocks: 2
        })
        .is_err());
}

#[test]
fn the_anchor_counts_in_the_entry_bytes() {
    let with = entry_of(ids(700), Some(512), Vec::new());
    let without = entry_of(ids(700), None, Vec::new());
    let anchor_bytes = with.anchor.as_ref().map(|a| {
        a.lin_caches
            .iter()
            .map(LinearAttnCache::resident_bytes)
            .sum::<u64>()
    });
    assert!(anchor_bytes.is_some_and(|b| b > 0));
    assert_eq!(Some(with.kv_bytes() - without.kv_bytes()), anchor_bytes);
}

/// A cache built as the arch builds its own, holding `stored` as its one slot.
fn arch_holding(stored: Qwen35MoeEntry) -> ArchPromptCache<Qwen35MoeEntry> {
    arch_holding_all(vec![stored])
}

/// The same, holding every entry of `stored`, in that slot order.
#[allow(
    clippy::unwrap_used,
    reason = "test-only: the cache is installed inside the same closure just above the pushes"
)]
fn arch_holding_all(stored: Vec<Qwen35MoeEntry>) -> ArchPromptCache<Qwen35MoeEntry> {
    let arch = ArchPromptCache::new(
        "test-qwen35",
        PROMPT_CACHE.policy(),
        crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS,
    );
    arch.with_inner_mut(|guard| {
        *guard = Some(PromptCache::new(4));
        for entry in stored {
            guard.as_mut().unwrap().push(entry);
        }
    });
    arch
}

fn consume(arch: &ArchPromptCache<Qwen35MoeEntry>, prompt: &[u32]) -> Consumed<Qwen35MoeEntry> {
    arch.consume(prompt, TEST_QUANT, TEST_LAYERS, false, TEST_SIG)
}

fn refusal(what: &str) -> Error {
    Error::Model(format!("expected {what}"))
}

#[test]
fn the_engine_resumes_a_ram_entry_at_its_anchor() -> Result<()> {
    let stored = entry(700);
    let prompt = diverging_at(&stored.prompt_token_ids, 699, 60);
    let arch = arch_holding(stored);
    let Consumed::Reuse { entry, kind } = consume(&arch, &prompt) else {
        return Err(refusal("an anchor resume"));
    };
    assert_eq!(
        kind,
        ReuseKind::BlockTruncate {
            effective_blocks: 2
        }
    );
    assert_eq!(conv_width(&entry.lin_caches), Some(ANCHOR_STATE_WIDTH));
    Ok(())
}

#[test]
fn the_engine_resumes_a_ram_entry_at_its_whole_prompt_when_the_request_extends_it() -> Result<()> {
    let stored = entry(700);
    let prompt = [stored.prompt_token_ids.clone(), ids(40)].concat();
    let arch = arch_holding(stored);
    let Consumed::Reuse { kind, .. } = consume(&arch, &prompt) else {
        return Err(refusal("a strict-prefix resume"));
    };
    assert_eq!(kind, ReuseKind::StrictPrefix { prefix_len: 700 });
    Ok(())
}

#[test]
fn the_engine_still_serves_an_equal_prompt_as_exact() {
    let stored = entry(700);
    let prompt = stored.prompt_token_ids.clone();
    let arch = arch_holding(stored);
    assert!(matches!(consume(&arch, &prompt), Consumed::Exact(_)));
}

#[test]
fn the_engine_refuses_a_request_that_diverges_before_the_anchor() {
    let stored = entry(700);
    let prompt = diverging_at(&stored.prompt_token_ids, 300, 60);
    let arch = arch_holding(stored);
    assert!(
        matches!(consume(&arch, &prompt), Consumed::Miss(reason) if reason.label() == "non_reusable"),
        "the slot matches on its first block and the hook refuses it"
    );
}

/// A slot that matches more blocks but cannot be cut back to its anchor.
fn slot_that_refuses_its_anchor(ids: Vec<u32>) -> Qwen35MoeEntry {
    let anchor = Anchor::pos_for(ids.len());
    entry_of(ids, anchor, vec![kv_holding(300)])
}

#[test]
fn a_second_slot_serves_a_request_the_best_slot_declines() -> Result<()> {
    let long = ids(1000);
    // The request shares three blocks with `long` and two with `short`; it
    // extends `short` and diverges from `long` past its anchor at 768.
    let request = [head(&long, 800), (0..60).map(|i| 1_000_000 + i).collect()].concat();
    let arch = arch_holding_all(vec![slot_that_refuses_its_anchor(long), entry(600)]);
    let Consumed::Reuse { kind, .. } = consume(&arch, &request) else {
        return Err(refusal("the second slot's strict-prefix resume"));
    };
    assert_eq!(kind, ReuseKind::StrictPrefix { prefix_len: 600 });
    Ok(())
}

#[test]
fn a_second_slot_that_equals_the_request_is_an_exact_hit() {
    // Two slots share two blocks with the request; the longer one is first and
    // cannot serve a request shorter than its anchor.
    let arch = arch_holding_all(vec![entry(800), entry(700)]);
    assert!(matches!(consume(&arch, &ids(700)), Consumed::Exact(_)));
}

#[test]
fn every_slot_declining_is_one_miss_with_the_best_slots_reason() {
    let long = ids(1000);
    let request = [head(&long, 800), (0..60).map(|i| 1_000_000 + i).collect()].concat();
    let arch = arch_holding_all(vec![
        slot_that_refuses_its_anchor(long.clone()),
        slot_that_refuses_its_anchor(head(&long, 900)),
    ]);
    assert!(
        matches!(consume(&arch, &request), Consumed::Miss(reason) if reason.label() == "non_reusable"),
        "no slot offers a state, so the request misses on the first slot's reason"
    );
}

#[test]
fn the_target_is_the_anchor_position_of_the_prompt_length() {
    assert_eq!(anchor_target(700, 0), Some(512));
    assert_eq!(anchor_target(257, 0), Some(256));
    assert_eq!(anchor_target(256, 0), None, "below one block of tail");
}

#[test]
fn a_resumed_request_captures_only_at_or_past_its_resume_point() {
    assert_eq!(anchor_target(740, 700), None, "the anchor 512 is behind it");
    assert_eq!(anchor_target(1300, 700), Some(1280));
    assert_eq!(anchor_target(769, 768), Some(768), "at the resume point");
}

#[test]
fn an_entry_is_stored_with_slots_and_without_paged_kv() {
    assert!(entry_is_storable(4, false));
    assert!(!entry_is_storable(0, false), "no slots");
    assert!(
        !entry_is_storable(4, true),
        "paged KV refuses a cache holding pages"
    );
}
