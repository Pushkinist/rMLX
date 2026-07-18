//! Integration test: `kv_cache_bytes` is sampled at ONE lifecycle point —
//! **post-decode** — on every arch, so the recorded number means the same thing
//! in every row of the metrics table.
//!
//! Gated behind `RMLX_KV_TEST_MODEL`. Point it at any served snapshot; the test
//! is arch-agnostic (it drives `Architecture::generate_greedy` and reads
//! `Architecture::kv_cache_bytes`, both dispatched per-arch):
//!
//!   RMLX_KV_TEST_MODEL=/path/to/model \
//!     cargo test -p rmlx-models --test kv_bytes_sample_point -- --ignored --nocapture
//!
//! Two assertions, neither of which trusts the metric's absolute value:
//!
//!   1. `kv_bytes_hit_equals_miss` — the exact-hit and cache-miss paths, on the
//!      SAME prompt + codec + shape, record the SAME figure. This is the sharp
//!      test: before the fix, the miss path sampled at the prefill snapshot
//!      (before the decode ring existed) while the exact-hit path sampled after
//!      decode, so the two disagreed by the ring on a ring-backed codec, and the
//!      recorded value depended on whether the prompt cache HIT.
//!
//!   2. `kv_bytes_grows_with_decode_length` — on a ring-backed codec
//!      (`IsoKOnly3`), a longer decode records MORE KV bytes than a shorter one
//!      (same prefill length, both cache-miss). A pre-decode sample is invariant
//!      to decode length — prefill is identical — so this delta collapses to
//!      zero if the sample point ever drifts back before decode. That makes it a
//!      re-drift guard: move the sample to pre-decode and this test goes red.
//!
//! `#[ignore]` so plain `cargo test` skips it (needs a real model + GPU).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::indexing_slicing,
    clippy::ignore_without_reason
)]

use std::path::PathBuf;

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

/// Ring-backed codec: `IsoKOnly3` stands up a GPU decode ring (no bf16
/// decode-seed early-return), so its resident footprint changes across the
/// decode phase — exactly the allocation a pre-decode sample cannot see.
const RING_CODEC: KvQuant = KvQuant::IsoKOnly3;

fn model_path() -> Option<PathBuf> {
    let Ok(p) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("RMLX_KV_TEST_MODEL not set — skipping kv_bytes_sample_point");
        return None;
    };
    Some(PathBuf::from(p))
}

/// Run one greedy generation and return the arch's recorded `kv_cache_bytes`.
///
/// `eos_ids` is empty so the decode loop always runs the full `n_tokens` steps
/// (no early EOS stop) — the byte figure is then a function of prefill length +
/// decode length only, which is what the two assertions compare.
fn generate_and_read_bytes(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    kv_quant: KvQuant,
    prompt_cache_slots: usize,
) -> u64 {
    let device = Device::Gpu;
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    model
        .generate_greedy(
            tokenizer,
            prompt_ids,
            n_tokens,
            device,
            Some(kv_quant),
            None, // max_ctx: arch default
            prompt_cache_slots,
            &[], // no EOS stop — force the full n_tokens decode steps
            &mut |_| None,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy");

    model.kv_cache_bytes()
}

/// Process resident-set size in bytes (macOS `ps -o rss=`, reported in KiB).
/// The #246/#258 anchor method: compare the reported KV bytes against the real
/// resident growth of the process, so the metric is tied to reality rather than
/// to its own formula.
fn process_rss_bytes() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<u64>()
        .expect("parse rss kib")
        * 1024
}

fn load() -> Option<(arch::Architecture, tokenizers::Tokenizer)> {
    let path = model_path()?;
    let device = Device::Gpu;
    let model =
        arch::load_model(&path, device, &arch::LoadOpts::default()).expect("arch::load_model");
    let tokenizer =
        tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).expect("tokenizer.json");
    Some((model, tokenizer))
}

/// The sharp test: exact-hit and cache-miss record the same lifecycle figure.
///
/// The prompt is padded past one prompt-cache block (`BLOCK_TOKENS = 256`) so the
/// second, identical request actually takes the EXACT-HIT path — a shorter prompt
/// yields no stored block and both calls silently miss, which would let this test
/// pass for the wrong reason. The `hits >= 1` assertion below makes that a hard
/// failure rather than a false green.
#[ignore]
#[test]
fn kv_bytes_hit_equals_miss() {
    let Some((model, tokenizer)) = load() else {
        return;
    };

    // >= 256 tokens so at least one full block is stored + matchable on repeat.
    let long_text = "The quick brown fox jumps over the lazy dog near the riverbank. ".repeat(48);
    let prompt_ids: Vec<u32> = tokenizer
        .encode(long_text, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    assert!(
        prompt_ids.len() >= 256,
        "prompt must exceed one 256-token block to be cache-matchable (got {})",
        prompt_ids.len()
    );

    let n_tokens = 48;

    // Cumulative-stat deltas (the prompt cache is a process-global static shared
    // with the other tests in this binary), so absolute counts are not portable.
    // `cache_stats()` is `None` until the first request initialises the cache;
    // treat that as a zero baseline.
    let stats = |m: &arch::Architecture| m.cache_stats().map_or((0, 0), |s| (s.hits, s.misses));
    let (_h0, m0) = stats(&model);

    // First call: cache MISS (fresh prompt) → stores a post-prefill snapshot and
    // records kv_cache_bytes post-decode.
    let miss_bytes =
        generate_and_read_bytes(&model, &tokenizer, &prompt_ids, n_tokens, RING_CODEC, 1);
    let (h1, m1) = stats(&model);

    // Second identical call: cache EXACT HIT → replays the snapshot, decodes the
    // same n_tokens, records kv_cache_bytes post-decode.
    let hit_bytes =
        generate_and_read_bytes(&model, &tokenizer, &prompt_ids, n_tokens, RING_CODEC, 1);
    let (h2, _m2) = stats(&model);

    println!(
        "[kv_bytes_hit_equals_miss] arch={} codec={RING_CODEC} prompt_len={} n_tokens={n_tokens} \
         miss={miss_bytes} hit={hit_bytes} miss_delta={} hit_delta={}",
        model.arch_class(),
        prompt_ids.len(),
        m1 - m0,
        h2 - h1,
    );

    // The first call must be a miss and the second an exact hit; otherwise the
    // equality below is trivially true (two post-decode misses) and proves nothing.
    assert_eq!(m1 - m0, 1, "first call must add exactly one cache miss");
    assert_eq!(
        h2 - h1,
        1,
        "second identical call must add exactly one cache hit — else the hit-vs-miss \
         comparison is vacuous"
    );

    assert!(miss_bytes > 0, "miss path recorded zero KV bytes");
    assert!(hit_bytes > 0, "hit path recorded zero KV bytes");
    assert_eq!(
        miss_bytes, hit_bytes,
        "kv_cache_bytes must be identical on the cache-miss and exact-hit paths \
         (same prompt + codec + shape) — a difference means the sample point still \
         depends on cache state"
    );
}

/// Reality anchor: the post-decode figure must be explained by real resident
/// memory, not conjured by a formula. Measure process RSS just after model load
/// (weights already resident) and again after a large-KV generation; the KV
/// growth the metric reports must fit inside the process's actual resident
/// growth and account for a dominant share of it.
#[ignore]
#[test]
fn kv_bytes_anchored_to_rss() {
    let Some((model, tokenizer)) = load() else {
        return;
    };

    // A long prompt so the KV footprint is a large, clearly-measurable share of
    // the resident growth (small KV would drown in allocator/page noise).
    let text = "In a distant valley the old cartographer traced every river and ridge. ".repeat(64);
    let prompt_ids: Vec<u32> = tokenizer
        .encode(text, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();

    // Baseline right after load: weights are resident, no KV yet. The whole
    // resident growth from here is the cost of serving this one request — KV,
    // its ring, decode scratch. `slots=0` keeps the prompt cache out of it so no
    // snapshot clone inflates the growth. The reported KV must fit inside it.
    let rss0 = process_rss_bytes();
    let kv_bytes = generate_and_read_bytes(&model, &tokenizer, &prompt_ids, 256, RING_CODEC, 0);
    let rss1 = process_rss_bytes();
    let rss_delta = rss1.saturating_sub(rss0);

    println!(
        "[kv_bytes_anchored_to_rss] arch={} codec={RING_CODEC} prompt_len={} kv_bytes={kv_bytes} \
         rss0={rss0} rss1={rss1} rss_delta={rss_delta} ratio={:.3}",
        model.arch_class(),
        prompt_ids.len(),
        kv_bytes as f64 / rss_delta.max(1) as f64,
    );

    assert!(kv_bytes > 0, "recorded zero KV bytes");
    assert!(
        rss_delta > 0,
        "generation did not grow resident memory — cannot anchor"
    );
    // Physically necessary: the reported KV cannot exceed the process's real
    // resident growth (plus slack for measurement granularity). A post-decode
    // over-report above real memory would be impossible.
    assert!(
        kv_bytes <= rss_delta + rss_delta / 4 + 64 * 1024 * 1024,
        "reported KV {kv_bytes} B exceeds real resident growth {rss_delta} B — the metric is \
         not anchored to memory"
    );
    // And it must be the same order of magnitude as the real growth (>= ~5%), so
    // the reported cache is genuinely resident, not a near-zero fabrication. The
    // exact share is size-dependent (small models spend most of a first-gen delta
    // on activation/buffer-pool scratch), so this is deliberately loose.
    assert!(
        kv_bytes * 20 >= rss_delta,
        "reported KV {kv_bytes} B is <5% of the resident growth {rss_delta} B — not \
         anchored to the real cache footprint"
    );
}

/// Ring/decode inclusion + re-drift guard: more decode = more recorded KV bytes.
///
/// Two cache-miss runs of equal prefill length but different decode length. The
/// only thing that can move the recorded figure is the decode phase, so a strict
/// inequality proves the sample is taken after decode. Under a pre-decode sample
/// the two are equal and this assertion fails.
#[ignore]
#[test]
fn kv_bytes_grows_with_decode_length() {
    let Some((model, tokenizer)) = load() else {
        return;
    };

    // Two distinct prompts (so each is a cache MISS, never a hit of the other),
    // sliced to a common length so their prefill KV footprint is identical — KV
    // size is a function of position count + shape, not token values.
    let a: Vec<u32> = tokenizer
        .encode(
            "Alpha bravo charlie delta echo foxtrot golf hotel india juliet.",
            true,
        )
        .expect("tokenize a")
        .get_ids()
        .to_vec();
    let b: Vec<u32> = tokenizer
        .encode(
            "Kilo lima mike november oscar papa quebec romeo sierra tango.",
            true,
        )
        .expect("tokenize b")
        .get_ids()
        .to_vec();
    let len = a.len().min(b.len());
    let prompt_short = &a[..len];
    let prompt_long = &b[..len];

    let n_short = 4;
    let n_long = 96;

    // Cache disabled-by-distinct-prompt: slots=1 but the two prompts differ, so
    // both are misses. Same prefill length → the delta is purely decode.
    let bytes_short =
        generate_and_read_bytes(&model, &tokenizer, prompt_short, n_short, RING_CODEC, 1);
    let bytes_long =
        generate_and_read_bytes(&model, &tokenizer, prompt_long, n_long, RING_CODEC, 1);

    println!(
        "[kv_bytes_grows_with_decode_length] arch={} codec={RING_CODEC} prefill_len={len} \
         short(n={n_short})={bytes_short} long(n={n_long})={bytes_long}",
        model.arch_class()
    );

    assert!(bytes_short > 0 && bytes_long > 0, "recorded zero KV bytes");
    assert!(
        bytes_long > bytes_short,
        "a longer decode ({n_long} tokens = {bytes_long} B) must record MORE KV bytes than a \
         shorter one ({n_short} tokens = {bytes_short} B) at equal prefill length — equality \
         means the sample was taken before decode, so the decode-time ring is invisible"
    );
}
