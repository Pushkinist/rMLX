use super::*;
use crate::prompt_cache::{PromptCache, SpillSink, BLOCK_TOKENS};
use rmlx_kv_ssd::chained_block_hashes;

/// Unit test: per-head q_norm shape is preserved.
///
/// Build a synthetic [1, seq=2, n_heads=4, head_dim=8] tensor,
/// apply RmsNorm with weight shape [head_dim], and confirm output shape
/// is unchanged. This exercises the "norm before transpose" path without
/// any snapshot or GPU dependency.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qwen3_per_head_qnorm_shape() {
    let batch = 1i32;
    let seq = 2i32;
    let n_heads = 4i32;
    let head_dim = 8i32;
    let n = (batch * seq * n_heads * head_dim) as usize;

    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let x =
        Array::from_f32_slice(&data, &[batch, seq, n_heads, head_dim]).expect("build input tensor");

    let w_data: Vec<f32> = vec![1.0_f32; head_dim as usize];
    let w = Array::from_f32_slice(&w_data, &[head_dim]).expect("build weight");

    let norm = RmsNorm {
        weight: w,
        eps: 1e-6,
    };
    let out = norm.forward(&x, Device::Cpu).expect("q_norm forward");
    out.eval().expect("eval");

    assert_eq!(
        out.shape(),
        vec![batch, seq, n_heads, head_dim],
        "per-head q_norm must preserve input shape"
    );
}

/// Integration test: load DR-Venus-4B snapshot (Qwen3, g64 b8 affine).
///
/// Skips if snapshot absent. Run explicitly:
/// cargo test -p rmlx-models integration_qwen3_dr_venus -- --ignored
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn integration_qwen3_dr_venus() {
    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_DR_VENUS").map(std::path::PathBuf::from)
    else {
        eprintln!("integration_qwen3_dr_venus: skipping: RMLX_TEST_MODEL_DR_VENUS not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        eprintln!("integration_qwen3_dr_venus: snapshot absent, skipping");
        return;
    }

    let model = load_from_path(model_dir, None).expect("load_from_path failed");

    let logits = model
        .forward_seq(&[151643], Device::Gpu)
        .expect("forward_seq failed");
    logits.eval().expect("logits eval");

    let vocab = model.cfg.vocab_size as i32;
    let logits_flat = logits.reshape(&[1, vocab], Device::Gpu).expect("reshape");
    logits_flat.eval().expect("logits_flat eval");

    assert_eq!(logits_flat.shape(), vec![1, vocab]);

    let lf32 = logits_flat.astype(Dtype::F32, Device::Cpu).expect("cast");
    lf32.eval().expect("f32 eval");
    let bytes = lf32.to_bytes().expect("to_bytes");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let nan_count = values.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "NaN in logits: {nan_count}");

    let max_logit = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_logit > 5.0 && max_logit < 100.0,
        "max_logit {max_logit:.2}"
    );
    eprintln!("qwen3 DR-Venus forward probe: max_logit={max_logit:.2}");
}

/// Integration test: load Ternary-Bonsai-8B snapshot (Qwen3, g128 b2 affine).
///
/// Exercises the 2-bit affine loader path. Skips if snapshot absent.
/// Run explicitly:
/// cargo test -p rmlx-models integration_qwen3_ternary_bonsai -- --ignored
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn integration_qwen3_ternary_bonsai() {
    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_BONSAI").map(std::path::PathBuf::from)
    else {
        eprintln!("integration_qwen3_ternary_bonsai: skipping: RMLX_TEST_MODEL_BONSAI not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        eprintln!("integration_qwen3_ternary_bonsai: snapshot absent, skipping");
        return;
    }

    let model = load_from_path(model_dir, None).expect("load_from_path failed");
    assert_eq!(model.cfg.quant_bits, 2, "expected 2-bit quantization");
    assert_eq!(model.cfg.quant_group_size, 128, "expected group_size=128");

    let logits = model
        .forward_seq(&[151643], Device::Gpu)
        .expect("forward_seq failed");
    logits.eval().expect("logits eval");

    let vocab = model.cfg.vocab_size as i32;
    let logits_flat = logits.reshape(&[1, vocab], Device::Gpu).expect("reshape");
    logits_flat.eval().expect("logits_flat eval");

    assert_eq!(logits_flat.shape(), vec![1, vocab]);

    let lf32 = logits_flat.astype(Dtype::F32, Device::Cpu).expect("cast");
    lf32.eval().expect("f32 eval");
    let bytes = lf32.to_bytes().expect("to_bytes");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let nan_count = values.iter().filter(|v| v.is_nan()).count();
    assert_eq!(nan_count, 0, "NaN in logits (2-bit): {nan_count}");

    let max_logit = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_logit > 5.0 && max_logit < 100.0,
        "max_logit {max_logit:.2}"
    );
    eprintln!("qwen3 Ternary-Bonsai 2-bit forward probe: max_logit={max_logit:.2}");
}

// -------------------------------------------------------------------------
// SSD spill/hydrate trait smoke tests for Qwen3Entry (pure-attention)
// -------------------------------------------------------------------------

/// Prove that `SpillSink<Qwen3Entry>` skips entries with no full block (no
/// stable spill key). CPU-only, no GPU, no disk.
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn qwen3_spill_sink_skips_entry_with_no_full_block() {
    // An entry whose prompt is shorter than one BLOCK_TOKENS → no block hash →
    // spill must be a no-op. We exercise this by routing a mock entry through
    // the prompt cache's slot-count eviction with a mock `SpillSink` that
    // records spilled hashes.
    use std::sync::{Arc, Mutex};

    struct HashSink {
        captured: Arc<Mutex<Vec<u64>>>,
    }
    impl SpillSink<Qwen3Entry> for HashSink {
        #[allow(
            clippy::clone_on_ref_ptr,
            reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
        )]
        #[allow(
            clippy::unwrap_used,
            reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
        )]
        fn spill(&self, e: &Qwen3Entry) {
            // Direct mirror of the `SsdSpiller::spill` skip-guard.
            if let Some(&h) = e.block_hashes.last() {
                self.captured.lock().unwrap().push(h);
            }
            // else: no block → skip, just like the real impl.
        }
    }

    let captured = Arc::new(Mutex::new(Vec::<u64>::new()));
    let mut cache: PromptCache<Qwen3Entry> = PromptCache::with_max_bytes(1, u64::MAX);
    cache.set_spill_sink(Box::new(HashSink {
        captured: captured.clone(),
    }));

    // Entry A: shorter than BLOCK_TOKENS → no block hash.
    let short_ids: Vec<u32> = (0..10u32).collect();
    cache.push(Qwen3Entry {
        prompt_token_ids: short_ids.clone(),
        block_hashes: chained_block_hashes(&short_ids),
        kv_caches: Vec::new(),
        first_id: 0,
        first_piece: String::new(),
        first_logprobs: None,
        kv_quant: Some(KvQuant::K8V8),
        is_ssd_hydrated: false,
    });

    // Entry B: also short, triggers slot eviction of A.
    let other_ids: Vec<u32> = (100..110u32).collect();
    cache.push(Qwen3Entry {
        prompt_token_ids: other_ids.clone(),
        block_hashes: chained_block_hashes(&other_ids),
        kv_caches: Vec::new(),
        first_id: 0,
        first_piece: String::new(),
        first_logprobs: None,
        kv_quant: Some(KvQuant::K8V8),
        is_ssd_hydrated: false,
    });

    // A was evicted but its block_hashes was empty → sink captured nothing.
    assert!(
        captured.lock().unwrap().is_empty(),
        "spill sink must skip entries with no full block"
    );
}

/// Prove that `SsdHydrate<Qwen3Entry>` round-trips: a mock hydrate source
/// that returns a pre-built `Qwen3Entry` is accepted by `hydrate_from_ssd`,
/// promoted into the RAM cache, and then found by `find_best_prefix`.
///
/// This exercises the `PromptCache<Qwen3Entry>::hydrate_from_ssd` + the
/// impl's field reconstruction (pure-attention: `lin_caches` discarded).
/// CPU-only, no GPU, no disk.
#[test]
#[allow(
    clippy::clone_on_ref_ptr,
    reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
)]
fn qwen3_ssd_hydrate_promotes_entry_into_ram() {
    use std::sync::{atomic::AtomicU64, Arc};

    /// Mock SSD source: returns a prebuilt `Qwen3Entry` if the prompt
    /// covers at least one full block, else a miss.
    struct MockSrc {
        calls: Arc<AtomicU64>,
        ids: Vec<u32>,
    }
    impl SsdHydrate<Qwen3Entry> for MockSrc {
        #[allow(
            clippy::clone_on_ref_ptr,
            reason = "intentional Arc::clone — explicit form aids grep for shared-ownership transfer sites"
        )]
        fn hydrate(&self, prompt_ids: &[u32]) -> Result<Option<Qwen3Entry>> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if prompt_ids.len() < BLOCK_TOKENS {
                return Ok(None);
            }
            let block_hashes = chained_block_hashes(&self.ids);
            Ok(Some(Qwen3Entry {
                prompt_token_ids: self.ids.clone(),
                block_hashes,
                kv_caches: Vec::new(),
                first_id: 0,
                first_piece: String::new(),
                first_logprobs: None,
                kv_quant: Some(KvQuant::K8V8),
                // Mirrors SsdHydrate::hydrate — hydrated entries are flagged.
                is_ssd_hydrated: true,
            }))
        }
    }

    let calls = Arc::new(AtomicU64::new(0));
    // Prompt: exactly 2 full blocks so there is a stable block hash.
    let prompt_ids: Vec<u32> = (0..(2 * BLOCK_TOKENS) as u32).collect();
    let mut cache: PromptCache<Qwen3Entry> = PromptCache::new(4);
    cache.set_ssd_source(Box::new(MockSrc {
        calls: calls.clone(),
        ids: prompt_ids.clone(),
    }));

    // Before hydrate: RAM miss.
    let before = cache.find_best_prefix(&prompt_ids, FNV_OFFSET);
    assert!(before.is_none(), "RAM must be empty before hydrate");

    // Hydrate from mock SSD.
    let promoted = cache.hydrate_from_ssd(&prompt_ids);
    assert!(
        promoted.is_some(),
        "mock SSD source must return a hit for a 2-block prompt"
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "hydrate called exactly once"
    );
    assert_eq!(cache.stats().ssd_hits, 1, "ssd_hits counter must be bumped");

    // After hydrate: RAM hit.
    let after = cache.find_best_prefix(&prompt_ids, FNV_OFFSET);
    assert!(
        after.is_some(),
        "promoted entry must be findable in RAM after hydrate"
    );
}

/// / : KV-bytes atomic store/read round-trip via the
/// unified `ArchPromptCache` last-bytes counter. Exercises the two store
/// sites: the Miss path (post-prefill snapshot push in `generate_greedy`)
/// and the Exact-hit path (added by , before `return Ok(steps)`).
/// CPU-only — no model, no GPU.
#[test]
fn qwen3_kv_bytes_store_read_roundtrip() {
    // Simulate the Miss path store.
    let sentinel_miss: u64 = 123_456_789;
    QWEN3_PROMPT_CACHE.store_kv_cache_bytes(sentinel_miss);
    assert_eq!(
        read_kv_cache_bytes(),
        sentinel_miss,
        "read_kv_cache_bytes() must return the value stored on the Miss path"
    );

    // Simulate the Exact-hit path store (same atomic, same semantics).
    let sentinel_hit: u64 = 987_654_321;
    QWEN3_PROMPT_CACHE.store_kv_cache_bytes(sentinel_hit);
    assert_eq!(
        read_kv_cache_bytes(),
        sentinel_hit,
        "read_kv_cache_bytes() must return the value stored on the Exact-hit path"
    );

    // Reset to 0 so parallel tests see a clean state.
    QWEN3_PROMPT_CACHE.store_kv_cache_bytes(0);
}

/// qwen3 dense is on the `ExactOnly` policy.
#[test]
fn qwen3_arch_policy_is_exact_only() {
    assert_eq!(QWEN3_PROMPT_CACHE.policy(), ReusePolicy::ExactOnly);
}

/// Regression: an SSD-hydrated full-prompt entry (block-aligned, no tail) must
/// NOT be served via the Exact fast path. Its `first_id` / `first_piece` are
/// placeholders (id 0) — replaying them emits a sentinel first token and seeds
/// decode with garbage. The `!is_ssd_hydrated` Exact-arm guard forces a
/// fall-through to a full re-prefill that re-derives the real first token.
///
/// Test structure (mirrors the qwen3_5_moe `hydrated_exact_block_no_tail`
/// regression):
///
/// 1. COLD: `generate_greedy` from an empty cache with a block-aligned prompt
///    (exact multiple of BLOCK_TOKENS, no tail) → record N_DECODE token ids.
/// 2. WARM: inject a real KV snapshot of the SAME full prompt, marked
///    `is_ssd_hydrated=true` with `first_id=0, first_piece=""` (exactly what
///    `SsdHydrate::hydrate` produces). Before the fix this is served as Exact →
///    `warm[0] == 0`. After the fix it falls to Miss → `warm == cold`.
///
/// Run:
/// ```sh
/// RMLX_TEST_MODEL_BONSAI=/path/to/Ternary-Bonsai-8B-mlx-2bit \
/// cargo test -p rmlx-models qwen3_hydrated_exact_no_tail_not_placeholder \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free; remaining unwrap is on values constructed in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length validated before call"
)]
fn qwen3_hydrated_exact_no_tail_not_placeholder() {
    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_BONSAI").map(std::path::PathBuf::from)
    else {
        println!("SKIP: RMLX_TEST_MODEL_BONSAI not set");
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
    if arch_str != "Qwen3ForCausalLM" {
        println!("SKIP: arch \"{arch_str}\" is not Qwen3ForCausalLM");
        return;
    }

    println!("Loading model from {}", model_dir.display());
    let model = load_from_path(model_dir, None).expect("load model");
    let n_layers = model.cfg.num_hidden_layers;
    let device = Device::Gpu;

    // Unquantized KV: COLD and WARM must produce token-identical output (the
    // merged none-payload spill fix restores KV exactly, so only the sentinel
    // would differ pre-fix).
    let kv_quant = KvQuant::None;
    let max_seq = 4096i32;

    // Prompt: exactly BLOCK_TOKENS = 256 tokens — NO tail (the SSD exact-hit
    // shape: the spilled block equals the full prompt).
    let prompt_len = BLOCK_TOKENS; // 256
    assert_eq!(
        prompt_len % BLOCK_TOKENS,
        0,
        "fixture must be block-aligned (no tail)"
    );
    let prompt_ids: Vec<u32> = (1u32..=prompt_len as u32)
        .map(|i| (i % 9999).max(1))
        .collect();

    let sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();
    let n_decode = 4usize;
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("load tokenizer");

    // ── Step 1: COLD full prefill ─────────────────────────────────────────────
    ensure_qwen3_prompt_cache(4);
    QWEN3_PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });

    let cold_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizer,
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("cold generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("COLD tokens: {cold_tokens:?}");
    assert_eq!(cold_tokens.len(), n_decode);
    assert_ne!(
        cold_tokens[0], 0,
        "cold first token is 0 — model anomaly; the regression check would be vacuous"
    );

    // ── Step 2: Build a real KV snapshot for the full block-aligned prompt ──
    let full_kv_caches = {
        use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
        let mut kv_caches: Vec<KvCache> = (0..n_layers)
            .map(|i| {
                let q = kv_quant_for_layer(
                    i,
                    n_layers,
                    kv_quant,
                    LAYER_ADAPTIVE_TAIL_N,
                    LAYER_ADAPTIVE_HEAD_N,
                );
                KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
            })
            .collect();
        for c in &mut kv_caches {
            c.enter_prefill();
        }
        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3");
        let n_chunks = prompt_len.div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            let logits = model
                .forward_seq_with_cache(chunk, Some(&mut kv_caches), device)
                .expect("prefill chunk");
            if is_last {
                logits.eval().expect("eval last-chunk logits");
            } else {
                for c in &kv_caches {
                    c.eval_prefill_state().expect("eval_prefill_state");
                }
            }
        }
        for c in &mut kv_caches {
            c.exit_prefill(device).expect("exit_prefill");
        }
        for c in &kv_caches {
            c.eval_for_spill().expect("eval_for_spill kv");
        }
        kv_caches
    };

    // ── Step 3: WARM — inject SSD-hydrated FULL-PROMPT snapshot ──────────────
    // first_id=0, first_piece="" — the placeholder a real SsdHydrate::hydrate
    // sets. Before the fix this entry is served as Exact → warm[0] == 0.
    QWEN3_PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
            let block_hashes =
                chained_block_hashes_seeded(&prompt_ids, FNV_OFFSET ^ kv_quant.cache_key_salt());
            let kv_snap: Result<Vec<_>> =
                full_kv_caches.iter().map(KvCache::try_deep_clone).collect();
            let kv_snap = kv_snap.expect("kv clone");
            cache.push(Qwen3Entry {
                prompt_token_ids: prompt_ids.clone(),
                block_hashes,
                kv_caches: kv_snap,
                first_id: 0,
                first_piece: String::new(),
                first_logprobs: None,
                kv_quant: Some(kv_quant),
                // KEY: full-length entry marked hydrated (no tail). Before the
                // fix the Exact arm fires → emits first_id=0. After the fix the
                // Exact arm skips → Miss → re-prefill.
                is_ssd_hydrated: true,
            });
        }
    });

    let warm_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizer,
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("warm generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("WARM tokens: {warm_tokens:?}");

    assert_ne!(
        warm_tokens.first().copied(),
        Some(0u32),
        "REGRESSION: warm first token is placeholder 0. \
         The Exact arm must be guarded by !is_ssd_hydrated."
    );
    assert_eq!(
        warm_tokens, cold_tokens,
        "regression: WARM must equal COLD after fix.\n\
         COLD: {cold_tokens:?}\n\
         WARM: {warm_tokens:?}"
    );
    println!(
        "PASS: SSD exact-hit sentinel guard works — warm[0]={} (not 0), warm==cold",
        warm_tokens[0]
    );
}

/// Phase A consume-engine migration golden (qwen3 dense, Bonsai).
///
/// Pins that routing qwen3 through the shared `consume()` engine is
/// behavior-identical to the pre-migration inline `CacheLookup`. Captures the
/// decoded `token_id` sequence at temp 0 for the four reachable cases and
/// asserts each against the cold baseline:
///   (i)   SSD exact-hit recompute: a hydrated FULL-prompt entry (no tail,
///         placeholder first_id) must fall to Miss → WARM == COLD (engine drops
///         it via the `!is_ssd_hydrated` Exact exclusion → re-prefill).
///   (ii)  RAM exact-hit: a real RAM snapshot of the same prompt replays the
///         stored first token → WARM == COLD.
///   (iii) first_logprobs replay: a RAM exact-hit with `top_logprobs_k = 3`
///         emits exactly one logprobs entry for the replayed first token,
///         truncated to 3 alternatives — the Miss path's capture survives the
///         Exact(E) clone.
///   (iv)  Miss: a distinct prompt re-prefills → coherent, non-placeholder
///         first token, distinct from the baseline.
///
/// Run:
/// ```sh
/// RMLX_TEST_MODEL_BONSAI=/path/to/Ternary-Bonsai-8B-mlx-2bit \
/// cargo test -p rmlx-models qwen3_consume_engine_migration_golden \
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
    reason = "test-only: a single golden covering the four reachable consume cases (cold/RAM/SSD/Miss) reads clearest as one sequential fixture"
)]
fn qwen3_consume_engine_migration_golden() {
    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_BONSAI").map(std::path::PathBuf::from)
    else {
        println!("SKIP: RMLX_TEST_MODEL_BONSAI not set");
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
    if arch_str != "Qwen3ForCausalLM" {
        println!("SKIP: arch \"{arch_str}\" is not Qwen3ForCausalLM");
        return;
    }

    let model = load_from_path(model_dir, None).expect("load model");
    let n_layers = model.cfg.num_hidden_layers;
    let device = Device::Gpu;
    let kv_quant = KvQuant::None;
    let max_seq = 4096i32;
    let n_decode = 6usize;
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("load tokenizer");

    // Block-aligned prompt (no tail) — the SSD exact-hit shape (the spilled
    // block equals the full prompt).
    let prompt_len = BLOCK_TOKENS;
    let prompt_ids: Vec<u32> = (1u32..=prompt_len as u32)
        .map(|i| (i % 9999).max(1))
        .collect();

    // A distinct block-aligned prompt for the Miss case.
    let other_ids: Vec<u32> = (1u32..=prompt_len as u32)
        .map(|i| ((i * 7 + 3) % 9999).max(1))
        .collect();

    let base_sampler = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();

    // Helper: run generate_greedy at temp 0 and return the decoded steps.
    let run = |ids: &[u32],
               sampler: &crate::sampler::SamplerConfig|
     -> Vec<crate::decode_loop::ProbeStep> {
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
            sampler,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy")
    };

    let clear_cache = || {
        ensure_qwen3_prompt_cache(4);
        QWEN3_PROMPT_CACHE.with_inner_mut(|guard| {
            if let Some(cache) = guard.as_mut() {
                cache.clear();
            }
        });
    };

    // ── (a) COLD baseline (Miss path: full prefill + store-back) ─────────────
    clear_cache();
    let cold: Vec<u32> = run(&prompt_ids, &base_sampler)
        .into_iter()
        .map(|s| s.token_id)
        .collect();
    println!("COLD tokens: {cold:?}");
    assert_eq!(cold.len(), n_decode);
    assert_ne!(
        cold[0], 0,
        "cold first token is the placeholder 0 — fixture anomaly"
    );

    // ── (ii) RAM exact-hit: the store-back from (a) is now resident. Re-running
    // the same prompt must be an Exact hit → token-identical to COLD. ─────────
    let warm_ram: Vec<u32> = run(&prompt_ids, &base_sampler)
        .into_iter()
        .map(|s| s.token_id)
        .collect();
    println!("WARM (RAM exact-hit) tokens: {warm_ram:?}");
    assert_eq!(
        warm_ram, cold,
        "RAM exact-hit must be token-identical to the cold baseline"
    );

    // ── (iii) first_logprobs truncated replay: a RAM exact-hit with
    // top_logprobs_k = 3 must emit exactly one logprobs entry for the replayed
    // first token (captured at store time, truncated to 3). ──────────────────
    let lp_sampler = crate::sampler::SamplerConfig {
        top_logprobs_k: 3,
        ..base_sampler
    };
    let lp_steps = run(&prompt_ids, &lp_sampler);
    let first = &lp_steps[0];
    let lp = first
        .logprobs
        .as_ref()
        .expect("exact-hit first token must carry replayed logprobs when lp_k > 0");
    assert_eq!(
        lp.token_id, cold[0],
        "replayed logprobs must describe the replayed first token"
    );
    assert!(
        lp.top.len() <= 3,
        "first_logprobs must be truncated to the request's top_logprobs_k (3), got {}",
        lp.top.len()
    );

    // ── (i) SSD exact-hit recompute: inject a hydrated FULL-prompt
    // snapshot (placeholder first_id=0, is_ssd_hydrated=true). The consume
    // engine's `!is_ssd_hydrated` Exact exclusion + the reuse hook declining the
    // equal-length case must drop it to Miss → WARM == COLD (never replay 0). ──
    let full_kv_caches = {
        use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
        let mut kv_caches: Vec<KvCache> = (0..n_layers)
            .map(|i| {
                let q = kv_quant_for_layer(
                    i,
                    n_layers,
                    kv_quant,
                    LAYER_ADAPTIVE_TAIL_N,
                    LAYER_ADAPTIVE_HEAD_N,
                );
                KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
            })
            .collect();
        for c in &mut kv_caches {
            c.enter_prefill();
        }
        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3");
        let n_chunks = prompt_len.div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            let logits = model
                .forward_seq_with_cache(chunk, Some(&mut kv_caches), device)
                .expect("prefill chunk");
            if is_last {
                logits.eval().expect("eval last-chunk logits");
            } else {
                for c in &kv_caches {
                    c.eval_prefill_state().expect("eval_prefill_state");
                }
            }
        }
        for c in &mut kv_caches {
            c.exit_prefill(device).expect("exit_prefill");
        }
        for c in &kv_caches {
            c.eval_for_spill().expect("eval_for_spill kv");
        }
        kv_caches
    };

    clear_cache();
    QWEN3_PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            let block_hashes =
                chained_block_hashes_seeded(&prompt_ids, FNV_OFFSET ^ kv_quant.cache_key_salt());
            let kv_snap: Result<Vec<_>> =
                full_kv_caches.iter().map(KvCache::try_deep_clone).collect();
            cache.push(Qwen3Entry {
                prompt_token_ids: prompt_ids.clone(),
                block_hashes,
                kv_caches: kv_snap.expect("kv clone"),
                first_id: 0,
                first_piece: String::new(),
                first_logprobs: None,
                kv_quant: Some(kv_quant),
                is_ssd_hydrated: true,
            });
        }
    });
    let warm_ssd: Vec<u32> = run(&prompt_ids, &base_sampler)
        .into_iter()
        .map(|s| s.token_id)
        .collect();
    println!("WARM (SSD exact-hit recompute) tokens: {warm_ssd:?}");
    assert_ne!(
        warm_ssd.first().copied(),
        Some(0u32),
        "SSD exact-hit must NOT replay the placeholder first_id 0"
    );
    assert_eq!(
        warm_ssd, cold,
        "SSD exact-hit recompute must equal the cold baseline"
    );

    // ── (iv) Miss: a distinct prompt re-prefills → coherent, distinct. ───────
    clear_cache();
    let miss: Vec<u32> = run(&other_ids, &base_sampler)
        .into_iter()
        .map(|s| s.token_id)
        .collect();
    println!("MISS (distinct prompt) tokens: {miss:?}");
    assert_eq!(miss.len(), n_decode);
    assert_ne!(miss[0], 0, "Miss first token must be a real decode token");

    println!(
        "PASS: consume-engine migration golden — RAM/SSD/logprobs/Miss all match the baseline"
    );
}

/// Dtype-lock for the YARN mscale multiply site.
///
/// When YARN is active (Bonsai runs factor ×4), q/k are scaled by the
/// precomputed `mscale` scalar before RoPE. The scalar is stored strong-f32
/// (the operand dtype is not known at construction). If it is multiplied into a
/// bf16 q/k as-is, MLX type-promotion widens q, k — and the K/V that reach the
/// cache — to f32, doubling KV residency on the `--kv-quant none` path. The fix
/// adopts the operand dtype before multiplying.
///
/// This test pins the invariant: a bf16 operand × dtype-adopted mscale stays
/// bf16, while a bf16 operand × raw strong-f32 mscale promotes to f32 (the bug
/// shape). A regression to the strong-f32 multiply makes this RED.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; an op error is the test failing"
)]
fn yarn_mscale_dtype_adopted_keeps_bf16() {
    // A bf16 q/k operand, as seen at the YARN multiply site.
    let operand = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4])
        .unwrap()
        .astype(Dtype::Bf16, Device::Cpu)
        .unwrap();

    // mscale is built strong-f32 at load (`scalar_f32(yarn_mscale)`).
    let mscale = scalar_f32(1.3);

    // Bug shape: multiplying the raw strong-f32 scalar promotes bf16 → f32.
    let promoted = multiply(&operand, &mscale, Device::Cpu).unwrap();
    promoted.eval().unwrap();
    assert_eq!(
        promoted.dtype(),
        Dtype::F32,
        "sanity: a raw strong-f32 mscale promotes a bf16 q/k to f32 (this is \
         the f32-KV-leak bug the multiply-site fix avoids)"
    );

    // Fix shape: the scalar adopts the operand dtype before multiplying.
    let adopted = mscale.astype(operand.dtype(), Device::Cpu).unwrap();
    let kept = multiply(&operand, &adopted, Device::Cpu).unwrap();
    kept.eval().unwrap();
    assert_eq!(
        kept.dtype(),
        Dtype::Bf16,
        "a dtype-adopted mscale must keep bf16 q/k at bf16 so the None-path KV \
         cache stores bf16, not f32"
    );
}

/// Dtype-lock for the RMSNorm weight.
///
/// The residual stream is bf16 (the embedding dequant is forced to bf16). Some
/// snapshots (e.g. Bonsai) ship norm weights at fp16. MLX's `rms_norm` promotes
/// a bf16 activation against an fp16 weight to f32 — and that f32 then
/// propagates through Q/K/V projections, attention, and the `--kv-quant none`
/// KV cache, doubling its residency. The load-time fix casts the norm weight to
/// bf16 so the norm output stays bf16.
///
/// This test pins the invariant directly on the `rms_norm` op: a bf16 input
/// normalized against an fp16 weight promotes to f32 (the bug shape), while a
/// bf16 weight keeps the output bf16 (the fix). A regression to loading the raw
/// fp16 weight makes this RED.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts on dtype; an op error is the test failing"
)]
fn rms_norm_bf16_weight_keeps_output_bf16() {
    // A bf16 residual-stream row, as the embedding produces it.
    let x = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4])
        .unwrap()
        .astype(Dtype::Bf16, Device::Cpu)
        .unwrap();

    // Bug shape: an fp16 norm weight (as shipped) promotes the output to f32.
    let w_f16 = Array::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4])
        .unwrap()
        .astype(Dtype::F16, Device::Cpu)
        .unwrap();
    let promoted = rms_norm(&x, Some(&w_f16), 1e-6, Device::Cpu).unwrap();
    promoted.eval().unwrap();
    assert_eq!(
        promoted.dtype(),
        Dtype::F32,
        "sanity: rms_norm(bf16 x, fp16 weight) promotes to f32 (this is the \
         f32-residual-stream leak the load-time bf16 cast avoids)"
    );

    // Fix shape: the norm weight is cast to bf16 at load.
    let w_bf16 = w_f16.astype(Dtype::Bf16, Device::Cpu).unwrap();
    let kept = rms_norm(&x, Some(&w_bf16), 1e-6, Device::Cpu).unwrap();
    kept.eval().unwrap();
    assert_eq!(
        kept.dtype(),
        Dtype::Bf16,
        "a bf16 norm weight must keep the bf16 residual stream at bf16, so the \
         None-path KV cache stores bf16, not f32"
    );
}
