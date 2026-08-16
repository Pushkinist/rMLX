use super::*;
use crate::prompt_cache::{PromptCache, ReuseKind, FNV_OFFSET};
use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::storage::KvStorage;
use rmlx_kv_quant::KvQuant;

fn entry_with(kv_caches: Vec<KvCache>, ids: Vec<u32>) -> Gemma4Entry {
    let block_hashes = rmlx_kv_ssd::chained_block_hashes(&ids);
    Gemma4Entry {
        prompt_token_ids: ids,
        block_hashes,
        kv_caches,
        first_id: 0,
        first_piece: String::new(),
        kv_quant: Some(KvQuant::K8V8),
        is_ssd_hydrated: false,
    }
}

fn entry_with_quant(
    kv_caches: Vec<KvCache>,
    ids: Vec<u32>,
    kv_quant: Option<KvQuant>,
) -> Gemma4Entry {
    let block_hashes = rmlx_kv_ssd::chained_block_hashes(&ids);
    Gemma4Entry {
        prompt_token_ids: ids,
        block_hashes,
        kv_caches,
        first_id: 0,
        first_piece: String::new(),
        kv_quant,
        is_ssd_hydrated: false,
    }
}

/// A freshly built snapshot — every cache at offset 0 — is trimmable, so
/// `can_truncate_to_block` permits the production block-truncate prefix
/// path. This is the cold-equal regime (cached prompt within
/// `sliding_window`, SWA ring not wrapped).
#[test]
fn can_truncate_when_no_swa_wrap() {
    let kv = vec![
        KvCache::with_quant_max_seq_window(KvQuant::K8V4, 8192, None), // full
        KvCache::with_quant_max_seq_window(KvQuant::K8V4, 8192, Some(1024)), // SWA
        KvCache::with_quant_max_seq_window(KvQuant::K8V4, 8192, Some(1024)), // SWA
    ];
    for c in &kv {
        assert!(c.is_trimmable(), "fresh cache (offset 0) must be trimmable");
    }
    let e = entry_with(kv, (0..512u32).collect());
    assert!(
        e.can_truncate_to_block(1),
        "all-trimmable snapshot must allow the block-truncate Prefix path"
    );
}

/// A flat (full-attention only) snapshot is always trimmable regardless of
/// length — mirrors mlx-lm `KVCache.is_trimmable()` -> True.
#[test]
fn flat_only_snapshot_always_trimmable() {
    let kv = vec![
        KvCache::with_quant_max_seq_window(KvQuant::K8V8, 8192, None),
        KvCache::with_quant_max_seq_window(KvQuant::Planar, 8192, None),
    ];
    let e = entry_with(kv, (0..256u32).collect());
    assert!(e.can_truncate_to_block(1));
}

/// MISMATCH path: stored `Some(K8V8)`, runtime `K8V4` → evict + miss.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn kv_quant_mismatch_evicts_and_misses() {
    let prompt_ids: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let entry = entry_with_quant(kv, prompt_ids.clone(), Some(KvQuant::K8V8));
    let mut cache: PromptCache<Gemma4Entry> = PromptCache::new(4);
    cache.push(entry);

    let runtime_quant = KvQuant::K8V4;
    let (slot_idx, _) = cache
        .find_best_prefix(&prompt_ids, FNV_OFFSET)
        .expect("block hit");
    let entry_quant = cache.slots[slot_idx].entry.kv_quant;
    assert_ne!(entry_quant, Some(runtime_quant));
    cache.evict_slot(slot_idx);

    assert_eq!(cache.slots.len(), 0);
    assert_eq!(cache.stats().evictions, 1);
    assert!(cache.find_best_prefix(&prompt_ids, FNV_OFFSET).is_none());
}

/// BACKWARD-COMPAT path: stored `None` — treated as MISMATCH.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn kv_quant_legacy_none_evicts_and_misses() {
    let prompt_ids: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let entry = entry_with_quant(kv, prompt_ids.clone(), None);
    let mut cache: PromptCache<Gemma4Entry> = PromptCache::new(4);
    cache.push(entry);

    let runtime_quant = KvQuant::K8V8;
    let (slot_idx, _) = cache
        .find_best_prefix(&prompt_ids, FNV_OFFSET)
        .expect("block hit");
    assert_ne!(cache.slots[slot_idx].entry.kv_quant, Some(runtime_quant));
    cache.evict_slot(slot_idx);
    assert_eq!(cache.slots.len(), 0);
    assert_eq!(cache.stats().evictions, 1);
}

/// MATCH path: stored `Some(K8V8)`, runtime `K8V8` → hit, no eviction.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn kv_quant_match_hits_no_eviction() {
    let prompt_ids: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let entry = entry_with_quant(kv, prompt_ids.clone(), Some(KvQuant::K8V8));
    let mut cache: PromptCache<Gemma4Entry> = PromptCache::new(4);
    cache.push(entry);

    let (slot_idx, _) = cache
        .find_best_prefix(&prompt_ids, FNV_OFFSET)
        .expect("block hit");
    assert_eq!(cache.slots[slot_idx].entry.kv_quant, Some(KvQuant::K8V8));
    assert_eq!(cache.slots.len(), 1);
    assert_eq!(cache.stats().evictions, 0);
}

/// gemma4's policy is `Partial` — the generate-loop is allowed to
/// take the block-aligned partial-prefix path (subject to the per-slot
/// `can_truncate_to_block` SWA gate).
#[test]
fn arch_policy_is_partial() {
    assert_eq!(PROMPT_CACHE.policy(), ReusePolicy::Partial);
}

/// A fully RAM-resident snapshot (every layer holds persistent K/V) is
/// hydrate-complete — the guard never degrades a normal RAM-cache hit.
#[test]
fn hydrate_complete_for_resident_snapshot() {
    let kv = vec![
        KvCache::with_quant_max_seq_window(KvQuant::K8V8, 8192, None), // full-attn
        KvCache::with_quant_max_seq_window(KvQuant::K8V8, 8192, Some(512)), // SWA
        KvCache::with_quant_max_seq_window(KvQuant::K8V8, 8192, Some(512)), // SWA
    ];
    let e = entry_with(kv, (0..(2 * BLOCK_TOKENS) as u32).collect());
    assert!(
        e.is_hydrate_complete(),
        "a RAM-resident snapshot must be hydrate-complete (no payload-less None layer)"
    );
}

/// An SSD-hydrated entry whose SWA layer came back as a payload-less
/// `KvStorage::None` (the rotating ring is not spilled) MUST be detected as
/// hydrate-INCOMPLETE so the generate loop degrades the prefix reuse to a full
/// re-prefill (Miss). The full-attention layer is reconstructed with real
/// quantized payload; the SWA layer is empty `None`.
#[test]
fn hydrate_incomplete_when_swa_layer_empty() {
    // Full-attention layer: real K8V8 storage with a recorded offset (the
    // hydrate path reconstructs these with payload).
    let full = KvCache::from_storage(
        KvStorage::new(KvQuant::K8V8, 8192),
        KvQuant::K8V8,
        256,
        0,
        DispatchPolicy::default(),
    );
    assert!(
        full.has_persistent_cache(),
        "K8V8 full-attn layer must report persistent cache"
    );
    // SWA layer: payload-less None (rotating ring dropped on spill), no bf16
    // seed restored — exactly what `block_io` reconstructs for a gemma4 SWA
    // layer on hydrate.
    let swa = KvCache::from_storage(
        KvStorage::None { max_seq: 512 },
        KvQuant::K8V8,
        256,
        1,
        DispatchPolicy::default(),
    );
    assert!(
        !swa.has_persistent_cache() && swa.decode_fp16_kv().is_none(),
        "dropped-SWA layer must be payload-less None with no bf16 seed"
    );

    let mut e = entry_with(vec![full, swa], (0..(2 * BLOCK_TOKENS) as u32).collect());
    // Scene-setting: model a hydrated entry. `is_hydrate_complete` inspects the
    // per-layer caches, not this flag, but a real degrade also requires it.
    e.is_ssd_hydrated = true;
    assert!(
        !e.is_hydrate_complete(),
        "a hydrated entry with an empty SWA None layer must be hydrate-INCOMPLETE \
         (excluded from the prefix reuse arms → Miss)"
    );
}

/// / B1: `is_strict_prefix_of` is the SWA snapshot/restore "hook"
/// (Risk #1 — silently breaking this would forfeit the wrapped-SWA
/// multi-turn flat-prefill path). Preserves token-identity, BLOCK_TOKENS
/// floor, and the strict-extension constraint (>= 1 new token).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn is_strict_prefix_of_swa_snapshot_hook() {
    let cached: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let e = entry_with(kv, cached.clone());

    // Strict extension by 1 token → hit.
    let mut ext = cached.clone();
    ext.push(999_999);
    assert!(
        e.is_strict_prefix_of(&ext),
        "strict extension by >= 1 token must be a hit"
    );

    // Equal length → NOT a strict prefix (Exact path handles this case).
    assert!(
        !e.is_strict_prefix_of(&cached),
        "equal-length prompt is not a strict prefix"
    );

    // Shorter request → NOT a strict prefix (caller must shrink instead).
    let short = &cached[..BLOCK_TOKENS];
    assert!(
        !e.is_strict_prefix_of(short),
        "shorter prompt is not a strict prefix of a longer cached entry"
    );

    // Below the BLOCK_TOKENS floor → NOT a strict prefix (worthwhile gate).
    let kv2 = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let short_cached: Vec<u32> = (0..(BLOCK_TOKENS - 1) as u32).collect();
    let e_short = entry_with(kv2, short_cached.clone());
    let mut ext_short = short_cached;
    ext_short.push(42);
    assert!(
        !e_short.is_strict_prefix_of(&ext_short),
        "cached_len < BLOCK_TOKENS must NOT be a strict prefix (worthwhile floor)"
    );

    // Divergent prompt → NOT a strict prefix.
    let mut diverged = cached.clone();
    diverged[0] = 99;
    diverged.push(123);
    assert!(
        !e.is_strict_prefix_of(&diverged),
        "divergent prompt is not a strict prefix"
    );
}

/// `is_reusable_prefix_of` block-truncate arithmetic (no model). The hook is
/// CONDITIONAL on whether the matched blocks cover the whole block-aligned
/// prompt:
///   - Block-aligned full match (`matched * 256 >= len`): drop the trailing
///     block so the re-prefilled tail is non-empty.
///       * A 3-block-aligned prompt matching all 3 blocks → `effective_blocks
///         == 2` (matched_blocks - 1).
///       * A single-block prompt matching its one block → `effective_blocks
///         == 0` → `None` (no tail blocks left).
///   - Genuine partial divergence (`matched * 256 < len`): reuse ALL matched
///     blocks; the tail past the last block boundary is already non-empty.
///       * A 512-token prompt diverging inside block 2 with `matched_blocks ==
///         1` → `BlockTruncate { effective_blocks: 1 }` (NOT `None`). The
///         pre-restoration unconditional `matched.min(prompt_blocks) - 1` would
///         have dropped this to `effective_blocks == 0` → `None` (a lost hit).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn is_reusable_prefix_of_block_truncate_arithmetic() {
    // Block-aligned 3-block prompt. A fresh (offset 0) flat cache is trimmable,
    // so `can_truncate_to_block` permits the block-truncate path. The cached
    // entry and the request are the SAME block-aligned prompt, so
    // `is_strict_prefix_of` is false (equal length, not a strict extension) and
    // the block-truncate arm runs.
    let block_count = 3usize;
    let ids: Vec<u32> = (0..(block_count * BLOCK_TOKENS) as u32).collect();
    let kv = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let e = entry_with(kv, ids.clone());
    assert!(
        !e.is_strict_prefix_of(&ids),
        "equal-length prompt must not take the strict-prefix arm — exercise block-truncate"
    );

    // matched_blocks == block_count == prompt_blocks; effective_blocks drops the
    // trailing block (block_count * 256 >= len) → block_count - 1.
    match e.is_reusable_prefix_of(&ids, false, block_count) {
        Some(ReuseKind::BlockTruncate { effective_blocks }) => {
            assert_eq!(
                effective_blocks,
                block_count - 1,
                "block-aligned full match must drop one trailing block"
            );
        }
        other => panic!("expected BlockTruncate, got {other:?}"),
    }

    // Single-block prompt matching its one block: effective_blocks saturates to
    // 0 → None (no tail blocks left → Miss).
    let single: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    let kv1 = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let e1 = entry_with(kv1, single.clone());
    assert!(
        !e1.is_strict_prefix_of(&single),
        "equal-length single-block prompt must not take the strict-prefix arm"
    );
    assert!(
        e1.is_reusable_prefix_of(&single, false, 1).is_none(),
        "all-blocks-consumed (effective_blocks == 0) must decline → None → Miss"
    );

    // Partial-divergence branch: a 512-token (2-block) request that shares only
    // its first FULL block with the cached entry (`matched_blocks == 1`) and
    // diverges inside block 2. `matched_blocks * 256 = 256 < 512`, so ALL
    // matched blocks are reused → `effective_blocks == 1`. The cached entry is a
    // distinct 512-token prompt (equal length, differing content) so the
    // strict-prefix arm declines and the block-truncate arm runs. A fresh
    // (offset 0) flat cache is trimmable, so `can_truncate_to_block(1)` holds.
    let request: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    let cached_ids: Vec<u32> = (1000..1000 + (2 * BLOCK_TOKENS) as u32).collect();
    let kv2 = vec![KvCache::with_quant_max_seq_window(
        KvQuant::K8V8,
        8192,
        None,
    )];
    let e2 = entry_with(kv2, cached_ids);
    assert!(
        !e2.is_strict_prefix_of(&request),
        "differing same-length cached prompt must not take the strict-prefix arm"
    );
    match e2.is_reusable_prefix_of(&request, false, 1) {
        Some(ReuseKind::BlockTruncate { effective_blocks }) => {
            assert_eq!(
                effective_blocks, 1,
                "partial divergence (matched * 256 < len) must reuse ALL matched blocks"
            );
        }
        other => panic!(
            "partial divergence with 1 shared block must be BlockTruncate{{1}}, \
             not a Miss; got {other:?}"
        ),
    }
}

/// Phase B consume-engine migration golden (gemma4, e2b — model-gated).
///
/// Pins that routing gemma4 through the shared `consume()` engine is
/// behavior-identical to the pre-migration inline dispatch across all
/// three reachable gemma4 paths. At temp 0, every reuse path must decode
/// token-identically to a cold (Miss) baseline of the SAME prompt:
///   (a) SWA degrade-to-reprefill: a 336-token (non-block-aligned) prompt whose
///       1 shared block is served from an SSD-hydrated entry with a payload-less
///       SWA layer (`is_hydrate_complete == false`) → the engine degrades it to
///       Miss → full re-prefill → WARM == COLD (the SWA-empty correctness fix).
///   (b) RAM multi-turn prefix reuse, BOTH arms:
///         - strict-prefix arm: a 512-token cached prompt that the request
///           extends to 768 tokens → `Reuse{StrictPrefix}` → restore + tail →
///           WARM == COLD(768);
///         - block-truncate arm: a 512-token cached prompt and a same-length
///           request that diverges in the second block (1 shared full block,
///           `matched * 256 < len`) → `Reuse{BlockTruncate{1}}` → trim + tail →
///           WARM == COLD(request).
///   (c) hydrated-exact-length exclusion: a block-aligned (256-token) SSD-hydrated entry whose
///       prefix length equals the full prompt (placeholder first_id 0) → the
///       `!is_ssd_hydrated` Exact exclusion drops it to Miss → recompute →
///       WARM == COLD, never the placeholder 0.
///
/// Run:
/// ```sh
/// RMLX_TEST_MODEL_GEMMA4_E2B=/path/to/mlx-community__gemma-4-e2b-it-mxfp8 \
/// cargo test -p rmlx-models gemma4_consume_engine_migration_golden \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: Mutex critical section is panic-free; remaining unwrap is on values constructed in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length validated before call"
)]
#[allow(
    clippy::too_many_lines,
    reason = "test-only: a single golden covering the three reachable gemma4 reuse paths (SWA-degrade / RAM strict-prefix + block-truncate / hydrated-exact-length exclusion) reads clearest as one sequential fixture"
)]
fn gemma4_consume_engine_migration_golden() {
    use crate::gemma4::{generate_greedy, load_from_path};
    use rmlx_mlx::Device;

    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E2B").map(std::path::PathBuf::from)
    else {
        println!("SKIP: RMLX_TEST_MODEL_GEMMA4_E2B not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: model dir not found at {}", model_dir.display());
        return;
    }
    let arch_str = {
        let cfg_path = model_dir.join("config.json");
        let data = std::fs::read(&cfg_path).expect("read config.json");
        let v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
        v.get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned()
    };
    if arch_str != "Gemma4ForConditionalGeneration" {
        println!("SKIP: arch \"{arch_str}\" is not Gemma4ForConditionalGeneration");
        return;
    }

    let model = load_from_path(model_dir).expect("load model");
    let device = Device::Gpu;
    let kv_quant = KvQuant::None;
    let max_seq = 4096i32;
    let n_decode = 6usize;
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("load tokenizer");

    // Deterministic synthetic token ids (kept in a safe mid-vocab band).
    let make_ids = |len: usize, salt: u32| -> Vec<u32> {
        (1u32..=len as u32)
            .map(|i| ((i.wrapping_mul(7).wrapping_add(salt)) % 9999).max(1))
            .collect()
    };

    let sampler = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();

    // Run generate_greedy at temp 0, return the decoded token_id sequence.
    let run = |ids: &[u32]| -> Vec<u32> {
        let mut rng = crate::sampler::Pcg32::new(sampler.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        generate_greedy(
            &model,
            &tokenizer,
            ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
            None,
        )
        .expect("generate_greedy")
        .into_iter()
        .map(|s| s.token_id)
        .collect()
    };

    let clear_cache = || {
        ensure_prompt_cache(4);
        PROMPT_CACHE.with_inner_mut(|guard| {
            if let Some(cache) = guard.as_mut() {
                cache.clear();
            }
        });
    };

    // Read back the RAM snapshot store-back left by the most recent cold run of
    // `ids`, deep-cloning its KV caches (the cache keeps ownership).
    let read_back_kv = |ids: &[u32]| -> Vec<KvCache> {
        let seed = crate::prompt_cache::cache_seed(active_layout_key(), kv_quant, model.model_sig);
        PROMPT_CACHE.with_inner_mut(|guard| {
            let cache = guard.as_mut().expect("cache present");
            let (slot_idx, _) = cache
                .find_best_prefix(ids, seed)
                .expect("cold store-back must be resident");
            cache.slots[slot_idx]
                .entry
                .kv_caches
                .iter()
                .map(|c| c.try_deep_clone().expect("kv clone"))
                .collect()
        })
    };

    // Push an SSD-hydrated entry (placeholder first_id 0, is_ssd_hydrated=true)
    // keyed on `key_ids`, carrying `kv_caches`, into a freshly cleared cache.
    let push_hydrated = |key_ids: &[u32], kv_caches: Vec<KvCache>| {
        clear_cache();
        PROMPT_CACHE.with_inner_mut(|guard| {
            let cache = guard.as_mut().expect("cache present");
            let block_hashes = crate::prompt_cache::chained_block_hashes_seeded(
                key_ids,
                crate::prompt_cache::cache_seed(active_layout_key(), kv_quant, model.model_sig),
            );
            cache.push(Gemma4Entry {
                prompt_token_ids: key_ids.to_vec(),
                block_hashes,
                kv_caches,
                first_id: 0,
                first_piece: String::new(),
                kv_quant: Some(kv_quant),
                is_ssd_hydrated: true,
            });
        });
    };

    // ── (a) SWA degrade-to-reprefill — 336-token (non-block-aligned) prompt ──
    // COLD baseline of the full 336-token prompt.
    let p336 = make_ids(336, 0);
    clear_cache();
    let cold_336 = run(&p336);
    assert_eq!(cold_336.len(), n_decode);
    assert_ne!(
        cold_336[0], 0,
        "cold first token is placeholder 0 — anomaly"
    );

    // Build a real snapshot of the first 256-token (1-block) prefix via a cold
    // run, then read its KV back and rebuild the SWA layers as payload-less
    // `KvStorage::None` — exactly what `block_io` reconstructs on hydrate (the
    // rotating ring is not serialised to the SSD tier).
    let prefix256: Vec<u32> = p336[..BLOCK_TOKENS].to_vec();
    clear_cache();
    let _ = run(&prefix256); // store-back leaves a RAM snapshot of the prefix
    let mut hydrated_kv = read_back_kv(&prefix256);
    for (i, c) in hydrated_kv.iter_mut().enumerate() {
        if matches!(
            model.cfg.layer_types[i],
            crate::gemma4::LayerType::SlidingAttention
        ) {
            // Drop the SWA ring to a payload-less None with no bf16 seed — the
            // incomplete-hydrate shape the engine must degrade.
            *c = KvCache::from_storage(
                KvStorage::None {
                    max_seq: model.cfg.sliding_window as i32,
                },
                kv_quant,
                BLOCK_TOKENS as i32,
                i,
                DispatchPolicy::default(),
            );
            assert!(
                !c.has_persistent_cache() && c.decode_fp16_kv().is_none(),
                "rebuilt SWA layer must be payload-less None"
            );
        }
    }
    push_hydrated(&prefix256, hydrated_kv);
    let warm_swa = run(&p336);
    println!("(a) SWA degrade: cold={cold_336:?} warm={warm_swa:?}");
    assert_ne!(
        warm_swa.first().copied(),
        Some(0u32),
        "SWA-incomplete hydrate must degrade to re-prefill, never replay placeholder 0"
    );
    assert_eq!(
        warm_swa, cold_336,
        "(a) SWA degrade-to-reprefill must equal the cold baseline"
    );

    // ── (b) RAM multi-turn prefix reuse — BOTH arms ─────────────────────────
    // Strict-prefix arm: cache a 512-token prompt, then extend to 768 tokens.
    let p512 = make_ids(512, 0);
    let p768: Vec<u32> = {
        let mut v = p512.clone();
        v.extend(make_ids(256, 5)); // 256-token tail with a distinct salt
        v
    };
    clear_cache();
    let cold_768 = run(&p768);
    clear_cache();
    let _ = run(&p512); // store-back caches the 512-token prompt (RAM)
    let warm_strict = run(&p768); // strict-prefix extension → Reuse{StrictPrefix}
    println!("(b) strict-prefix: cold={cold_768:?} warm={warm_strict:?}");
    assert_eq!(
        warm_strict, cold_768,
        "(b) RAM strict-prefix reuse must equal the cold baseline for the extended prompt"
    );

    // Block-truncate arm: a same-length 512-token request that shares the first
    // block then diverges in the second block. find_best_prefix matches 1 block;
    // `is_strict_prefix_of` is false (divergent, equal length) so the
    // block-truncate arm fires (effective_blocks = 1).
    let p512_div: Vec<u32> = {
        let mut v = p512.clone();
        for t in v.iter_mut().skip(BLOCK_TOKENS) {
            *t = ((*t).wrapping_add(101) % 9999).max(1);
        }
        v
    };
    clear_cache();
    let cold_div = run(&p512_div);
    clear_cache();
    let _ = run(&p512); // cache the 512-token prompt sharing the first block
    let warm_trunc = run(&p512_div); // 1 shared block, diverges → Reuse{BlockTruncate}
    println!("(b) block-truncate: cold={cold_div:?} warm={warm_trunc:?}");
    assert_eq!(
        warm_trunc, cold_div,
        "(b) RAM block-truncate reuse must equal the cold baseline for the divergent prompt"
    );

    // ── (c) hydrated-exact-length exclusion: block-aligned hydrated entry == full prompt ──
    // A 256-token block-aligned prompt; the hydrated entry's prefix length
    // equals the full prompt (no tail, placeholder first_id 0). The Exact arm's
    // `!is_ssd_hydrated` guard drops it to Miss → recompute → WARM == COLD.
    let p256 = make_ids(BLOCK_TOKENS, 13);
    clear_cache();
    let cold_256 = run(&p256);
    assert_ne!(
        cold_256[0], 0,
        "cold 256 first token is placeholder 0 — anomaly"
    );
    clear_cache();
    let _ = run(&p256); // build a real RAM snapshot of the full block-aligned prompt
    let full_kv = read_back_kv(&p256);
    push_hydrated(&p256, full_kv); // re-key as a hydrated full-length entry
    let warm_exact = run(&p256);
    println!("(c) hydrated-exact-length exclusion: cold={cold_256:?} warm={warm_exact:?}");
    assert_ne!(
        warm_exact.first().copied(),
        Some(0u32),
        "(c) block-aligned hydrated exact-length must NOT replay placeholder 0"
    );
    assert_eq!(
        warm_exact, cold_256,
        "(c) block-aligned hydrated exact-length recompute must equal the cold baseline"
    );

    println!(
        "PASS: gemma4 consume-engine migration golden — SWA-degrade / strict-prefix / \
         block-truncate / hydrated-exact-length exclusion all match their cold baselines"
    );
}
