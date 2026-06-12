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
