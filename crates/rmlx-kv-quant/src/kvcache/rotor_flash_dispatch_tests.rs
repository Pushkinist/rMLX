//! Production-path dispatch tests for the rotor flash-decode kernel.
//!
//! The oracle tests in `rotor_flash_decode_msl_tests.rs` prove the kernel is
//! numerically right when called directly. These prove the thing that actually
//! matters in production: that `KvCache::update_and_sdpa` **reaches** it on a
//! rotor K-only cache, instead of silently falling through to the O(seq) CPU
//! dequant path the kernel exists to remove.
//!
//! # No env dependence
//!
//! These tests never touch `RMLX_ROTOR_QJL`. The codec fixes its QJL decision
//! on the first append (`rotors.is_empty()` lazy-init) and never revisits it, so
//! seeding a store with a pre-built rotor table pins QJL for that store's
//! lifetime — deterministically, whatever any other test in the binary does to
//! the process-global env var.

use super::KvCache;
use crate::clifford::make_rotor_table;
use crate::quant::KvQuant;
use crate::rotor_flash_decode_msl::rotor_flash_decode_dispatch_count;
use crate::rotor_flash_decode_symv_msl::rotor_symv_flash_decode_dispatch_count;
use crate::rotorquant::{make_qjl_projection, n_groups_for};
use crate::storage::{KvStorage, QuantRotorK3, QuantRotorK4, QuantRotorV3, QuantRotorV4};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

const MAX_SEQ: i32 = 512;

/// What one `run_decode_steps` call observed.
struct DecodeProbe {
    /// Dispatch-count delta across the decode steps.
    ///
    /// The counter is process-global, so a concurrent test can only ever
    /// *inflate* this — `>=` assertions on it are race-free, `== 0` ones are
    /// not. Use `gpu_ring_live` for the negative case.
    delta: u64,
    /// Whether the rotor store's GPU ring is live at the end of the run.
    ///
    /// Local to this cache, so unlike `delta` it is immune to other tests. The
    /// flash path cannot run without the ring, so `!gpu_ring_live` is a
    /// race-free proof that decode stayed on the CPU dequant path.
    gpu_ring_live: bool,
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("f32_array")
}

/// Build a rotor K-only cache whose QJL decision is pinned to `use_qjl`,
/// independent of the process-global toggle.
///
/// Pre-seeding `rotors` is what pins it: both the CPU and GPU append paths only
/// touch `qjl_s_matrix` inside their `rotors.is_empty()` lazy-init branch, so a
/// store handed a non-empty rotor table keeps whatever `qjl_s_matrix` it was
/// built with.
fn seeded_cache(quant: KvQuant, kv_h: i32, head_dim: i32, use_qjl: bool) -> KvCache {
    seeded_cache_b(quant, 1, kv_h, head_dim, use_qjl)
}

/// [`seeded_cache`] with an explicit batch dimension.
fn seeded_cache_b(quant: KvQuant, b: i32, kv_h: i32, head_dim: i32, use_qjl: bool) -> KvCache {
    let n_groups = n_groups_for(head_dim as usize);
    let rotors = make_rotor_table(0, 0, n_groups);
    let qjl = use_qjl.then(|| make_qjl_projection(head_dim as usize));
    let shape = vec![b, kv_h, 0, head_dim];

    let storage = if quant == KvQuant::RotorKOnly4 {
        KvStorage::RotorKOnly4 {
            k: Some(QuantRotorK4::from_cpu_blocks(
                rotors,
                qjl,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        }
    } else {
        KvStorage::RotorKOnly3 {
            k: Some(QuantRotorK3::from_cpu_blocks(
                rotors,
                qjl,
                Vec::new(),
                shape,
                0,
            )),
            max_seq: MAX_SEQ,
        }
    };
    KvCache::from_storage(storage, quant, 0, 0, DispatchPolicy::default(), false)
}

/// Drive one prefill chunk + 4 decode steps through the production
/// `update_and_sdpa` entry point.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn run_decode_steps(
    quant: KvQuant,
    kv_h: i32,
    n_q_heads: i32,
    head_dim: i32,
    prefill: i32,
    use_qjl: bool,
) -> DecodeProbe {
    let device = Device::Gpu;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut cache = seeded_cache(quant, kv_h, head_dim, use_qjl);

    // Prefill chunk (q_seq > 1): the flash path is decode-only, so this goes
    // through the legacy path and leaves an accumulated prefix behind —
    // exactly the state the first decode step must seed the GPU ring from.
    let pf = (prefill * kv_h * head_dim) as usize;
    let k = f32_array(&lcg_data(pf, 1), &[1, kv_h, prefill, head_dim]);
    let v = f32_array(&lcg_data(pf, 2), &[1, kv_h, prefill, head_dim]);
    let q = f32_array(
        &lcg_data((prefill * n_q_heads * head_dim) as usize, 3),
        &[1, n_q_heads, prefill, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    let before = rotor_flash_decode_dispatch_count();
    for step in 0..4_u64 {
        let one = (kv_h * head_dim) as usize;
        let k1 = f32_array(&lcg_data(one, 10 + step), &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(&lcg_data(one, 20 + step), &[1, kv_h, 1, head_dim]);
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 30 + step),
            &[1, n_q_heads, 1, head_dim],
        );
        let out = cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa");
        out.eval().expect("decode out eval");
        assert_eq!(
            out.shape(),
            vec![1, n_q_heads, 1, head_dim],
            "decode output shape"
        );
    }
    let delta = rotor_flash_decode_dispatch_count() - before;
    let gpu_ring_live = rotor_k_ring_live(&cache);
    DecodeProbe {
        delta,
        gpu_ring_live,
    }
}

/// Whether the active rotor K-only store holds a live GPU ring.
fn rotor_k_ring_live(cache: &KvCache) -> bool {
    if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else {
        false
    }
}

/// Build a symmetric rotor cache with QJL pinned off, the same way
/// [`seeded_cache`] pins it for the K-only variants: a pre-seeded rotor table
/// means neither append path re-enters its `rotors.is_empty()` lazy-init, so
/// `qjl_s_matrix` stays `None` whatever the process-global toggle says.
fn seeded_sym_cache(quant: KvQuant, kv_h: i32, head_dim: i32) -> KvCache {
    let n_groups = n_groups_for(head_dim as usize);
    let rotors = make_rotor_table(0, 0, n_groups);
    let shape = vec![1, kv_h, 0, head_dim];

    let storage = if quant == KvQuant::Rotor4Sym {
        KvStorage::RotorSym4 {
            k: Some(QuantRotorK4::from_cpu_blocks(
                rotors,
                None,
                Vec::new(),
                shape.clone(),
                0,
            )),
            v: Some(QuantRotorV4::new(shape, MAX_SEQ, 0)),
            max_seq: MAX_SEQ,
        }
    } else {
        KvStorage::RotorSym3 {
            k: Some(QuantRotorK3::from_cpu_blocks(
                rotors,
                None,
                Vec::new(),
                shape.clone(),
                0,
            )),
            v: Some(QuantRotorV3::new(shape, MAX_SEQ, 0)),
            max_seq: MAX_SEQ,
        }
    };
    KvCache::from_storage(storage, quant, 0, 0, DispatchPolicy::default(), false)
}

/// `(cpu blocks, ring live)` for the K axis of the active symmetric rotor store.
fn sym_k_store_state(cache: &KvCache) -> (usize, bool) {
    if let KvStorage::RotorSym3 { k: Some(ks), .. } = cache.storage() {
        (ks.blocks.len(), ks.gpu.is_allocated())
    } else if let KvStorage::RotorSym4 { k: Some(ks), .. } = cache.storage() {
        (ks.blocks.len(), ks.gpu.is_allocated())
    } else {
        (0, false)
    }
}

/// CPU blocks the active rotor K-only store currently holds.
fn rotor_k_blocks_len(cache: &KvCache) -> usize {
    if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.blocks.len()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.blocks.len()
    } else {
        0
    }
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_3_decode_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 2, 8, 128, 24, false);
    assert!(
        probe.delta >= 4,
        "update_and_sdpa did not reach the rotor3 flash-decode kernel on all 4 \
         decode steps (delta={}) — the codec is still CPU-dequanting the whole \
         K prefix every step",
        probe.delta
    );
    assert!(probe.gpu_ring_live, "rotor3 GPU ring should be live");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_4_decode_dispatches_flash_kernel() {
    if skip_if_no_gpu_env() {
        return;
    }
    let probe = run_decode_steps(KvQuant::RotorKOnly4, 2, 8, 128, 24, false);
    assert!(
        probe.delta >= 4,
        "update_and_sdpa did not reach the rotor4 flash-decode kernel (delta={})",
        probe.delta
    );
    assert!(probe.gpu_ring_live, "rotor4 GPU ring should be live");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_decode_dispatches_across_a_tile_boundary() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Prefill past TILE_SIZE so the decode steps span multiple pass-1 tiles and
    // exercise the pass-2 merge on the production path.
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 1, 4, 128, 100, false);
    assert!(
        probe.delta >= 4,
        "multi-tile decode did not dispatch (delta={})",
        probe.delta
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_decode_dispatches_at_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Gemma3-class shape. The kernel keys off shape, not arch.
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 4, 8, 256, 24, false);
    assert!(
        probe.delta >= 4,
        "head_dim=256 decode did not dispatch (delta={})",
        probe.delta
    );
}

/// `b > 1` must skip the ring feed, not attempt (and fail) it.
///
/// `QuantKGpuRing`'s per-step stride is `kv_h * n_groups` and does not interleave
/// batch, so a batched chunk cannot be laid into it: the encode arrays carry
/// `b * kv_h * new_seq * n_groups` entries against a `new_seq * kv_h * n_groups`
/// span. Attempting it returns `Err` and kills the request, where the CPU blocks
/// (which handle `b > 1`) would have served it fine.
///
/// Drives the ring-maintaining entry point **directly** rather than through
/// `update_and_sdpa`: the fused SDPA dispatcher carries its own `b == 1` gate
/// that keeps the flash kernel away from a batched chunk, so an end-to-end call
/// would never reach the ring-feed skip this test exercises. Feeding the `b > 1`
/// chunk straight to the append entry point puts it directly on the skip path.
fn batched_ring_feed_is_skipped(quant: KvQuant, bits_label: &str) {
    let device = Device::Gpu;
    let (b, kv_h, head_dim, new_seq) = (2_i32, 2_i32, 128_i32, 4_i32);
    let mut cache = seeded_cache_b(quant, b, kv_h, head_dim, false);
    let shape = [b, kv_h, new_seq, head_dim];
    let k = f32_array(
        &lcg_data((b * kv_h * new_seq * head_dim) as usize, 7),
        &shape,
    );

    let res = if quant == KvQuant::RotorKOnly4 {
        super::update::rotor4_k_only_gpu_append(&mut cache, &k, &shape, device)
    } else {
        super::update::rotor3_k_only_gpu_append(&mut cache, &k, &shape, device)
    };
    res.unwrap_or_else(|e| {
        panic!("{bits_label}: batched GPU append must not error, got: {e}");
    });

    let ring_live = if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        ks.gpu.is_allocated()
    } else {
        panic!("{bits_label}: store vanished");
    };
    assert!(
        !ring_live,
        "{bits_label}: a b>1 cache must not carry a ring — the stride cannot represent it"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor3_batched_gpu_append_skips_the_ring_instead_of_erroring() {
    if skip_if_no_gpu_env() {
        return;
    }
    batched_ring_feed_is_skipped(KvQuant::RotorKOnly3, "rotor3");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor4_batched_gpu_append_skips_the_ring_instead_of_erroring() {
    if skip_if_no_gpu_env() {
        return;
    }
    batched_ring_feed_is_skipped(KvQuant::RotorKOnly4, "rotor4");
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_decode_stays_cpu_when_store_carries_qjl() {
    if skip_if_no_gpu_env() {
        return;
    }
    // The kernel cannot reproduce the QJL residual, so a QJL-carrying store must
    // NOT reach it — a dispatch here would mean decode silently dropped the
    // residual and changed the codec's numerics.
    let probe = run_decode_steps(KvQuant::RotorKOnly3, 2, 8, 128, 24, true);
    // Asserted on the cache's own ring, not the process-global dispatch
    // counter: a concurrently-running test dispatching the kernel would make a
    // `delta == 0` assertion flake. The flash path cannot run without this
    // ring, so its absence is the race-free proof.
    assert!(
        !probe.gpu_ring_live,
        "rotor GPU ring was populated on a QJL store — decode would read a store \
         missing the QJL residual"
    );
}

/// Global cosine over two flat vectors (0 when either is all-zeros).
fn mean_cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// `(dequant, shape, cpu_block_tokens, ring_live)` for the active rotor K store.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn rotor_k_store_probe(cache: &KvCache) -> (Vec<f32>, Vec<i32>, usize, bool) {
    // if-let chain (not a `match` with a wildcard arm) — matches the negative-case
    // helpers above and stays clear of `clippy::wildcard_enum_match_arm`.
    if let KvStorage::RotorKOnly3 { k: Some(ks), .. } = cache.storage() {
        (
            ks.dequant().expect("rotor3 dequant"),
            ks.shape.clone(),
            ks.blocks.iter().map(|b| b.n_tokens).sum(),
            ks.gpu.is_allocated(),
        )
    } else if let KvStorage::RotorKOnly4 { k: Some(ks), .. } = cache.storage() {
        (
            ks.dequant().expect("rotor4 dequant"),
            ks.shape.clone(),
            ks.blocks.iter().map(|b| b.n_tokens).sum(),
            ks.gpu.is_allocated(),
        )
    } else {
        panic!("expected a live rotor K-only store")
    }
}

/// Reference K reconstruction: feed the identical K tokens through a CPU-only
/// rotor K store (complete blocks, no ring) and dequant.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn cpu_reference_dequant(
    quant: KvQuant,
    kv_h: i32,
    head_dim: i32,
    k_prefill: &[f32],
    step_ks: &[Vec<f32>],
    prefill: i32,
) -> Vec<f32> {
    let n_groups = n_groups_for(head_dim as usize);
    let rotors = make_rotor_table(0, 0, n_groups);
    let shape0 = vec![1, kv_h, 0, head_dim];
    let pf_shape = [1, kv_h, prefill, head_dim];
    let step_shape = [1, kv_h, 1, head_dim];
    if quant == KvQuant::RotorKOnly4 {
        let mut ks = QuantRotorK4::from_cpu_blocks(rotors, None, Vec::new(), shape0, 0);
        ks.append(k_prefill, &pf_shape)
            .expect("ref rotor4 prefill append");
        for s in step_ks {
            ks.append(s, &step_shape).expect("ref rotor4 step append");
        }
        ks.dequant().expect("ref rotor4 dequant")
    } else {
        let mut ks = QuantRotorK3::from_cpu_blocks(rotors, None, Vec::new(), shape0, 0);
        ks.append(k_prefill, &pf_shape)
            .expect("ref rotor3 prefill append");
        for s in step_ks {
            ks.append(s, &step_shape).expect("ref rotor3 step append");
        }
        ks.dequant().expect("ref rotor3 dequant")
    }
}

/// Full-prefix correctness once the GPU ring is the store's only copy of K.
///
/// After prefill + N fused decode steps a rotor K-only store holds **nothing**
/// on the host: the per-step block download is skipped, and the append releases
/// the seeded prefill blocks as soon as the ring goes live, so the ring carries
/// the whole `[0, prefill + steps)` prefix on its own. `dequant()` must still
/// return the FULL prefix, rebuilt from the ring, with **no zero-padded gap**.
/// Proven against a CPU-only reference store fed the identical K tokens.
///
/// The empty-blocks precondition is what makes the comparison load-bearing: with
/// a host copy still resident, `dequant()` could pass by reading it and prove
/// nothing about the ring.
///
/// Mutation check: make `dequant()` decode `self.blocks` directly instead of
/// `synced_rotor_k_blocks(...)`. With no host blocks left it decodes nothing,
/// falls into the empty-store arm and returns a correctly-sized all-zero buffer
/// — the length guard is satisfied, and the cosine against the CPU reference
/// collapses to 0 (RED), catching exactly the zeroed prefix the ring rebuild
/// prevents.
#[allow(clippy::expect_used, reason = "test: invariants documented")]
fn ring_only_tail_dequant_is_full_prefix(quant: KvQuant) {
    let device = Device::Gpu;
    let (kv_h, n_q_heads, head_dim, prefill, steps) = (2_i32, 8_i32, 128_i32, 6_i32, 5_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Known K tokens: one prefill chunk + `steps` single-token decode chunks.
    let pf_n = (prefill * kv_h * head_dim) as usize;
    let k_prefill = lcg_data(pf_n, 101);
    let step_ks: Vec<Vec<f32>> = (0..steps)
        .map(|s| lcg_data((kv_h * head_dim) as usize, 201 + s as u64))
        .collect();

    // Fused cache: prefill + decode through the production entry point.
    let mut cache = seeded_cache(quant, kv_h, head_dim, false);
    let k = f32_array(&k_prefill, &[1, kv_h, prefill, head_dim]);
    let v = f32_array(&lcg_data(pf_n, 102), &[1, kv_h, prefill, head_dim]);
    let q = f32_array(
        &lcg_data((prefill * n_q_heads * head_dim) as usize, 103),
        &[1, n_q_heads, prefill, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");
    for (i, ks_step) in step_ks.iter().enumerate() {
        let k1 = f32_array(ks_step, &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(
            &lcg_data((kv_h * head_dim) as usize, 301 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 401 + i as u64),
            &[1, n_q_heads, 1, head_dim],
        );
        cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa")
            .eval()
            .expect("decode out eval");
    }

    let (fused_dq, shape, blocks_tokens, ring_live) = rotor_k_store_probe(&cache);
    let full_seq = prefill + steps;
    let full_len = (kv_h * full_seq * head_dim) as usize;

    // Ring-only precondition: the ring is live and is the store's sole copy of
    // K — the append released the seeded prefill blocks. Anything left on the
    // host would let the dequant below pass without touching the ring.
    assert!(ring_live, "ring must be live for the fused decode tail");
    assert_eq!(
        blocks_tokens,
        0,
        "CPU blocks must be released once the ring is live (ring is the sole \
         resident copy); got {blocks_tokens} tokens, shape[2]={}",
        shape.get(2).copied().unwrap_or(0)
    );
    assert_eq!(
        shape.get(2).copied().unwrap_or(0),
        full_seq,
        "shape[2] must advance with the ring"
    );

    // dequant returns the FULL prefix — no truncation, no zero-padded gap.
    assert_eq!(
        fused_dq.len(),
        full_len,
        "dequant must cover the full [1,{kv_h},{full_seq},{head_dim}] prefix"
    );
    let ref_dq = cpu_reference_dequant(quant, kv_h, head_dim, &k_prefill, &step_ks, prefill);
    assert_eq!(ref_dq.len(), fused_dq.len(), "reference length mismatch");

    let cos = mean_cosine(&fused_dq, &ref_dq);
    assert!(
        cos > 0.99,
        "ring-only-tail dequant vs CPU reference cosine {cos} too low — a zero-padded \
         or truncated tail would drop it well below this"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_3_ring_only_tail_dequant_is_full_prefix() {
    if skip_if_no_gpu_env() {
        return;
    }
    ring_only_tail_dequant_is_full_prefix(KvQuant::RotorKOnly3);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_4_ring_only_tail_dequant_is_full_prefix() {
    if skip_if_no_gpu_env() {
        return;
    }
    ring_only_tail_dequant_is_full_prefix(KvQuant::RotorKOnly4);
}

/// Truncating a LIVE rotor-K-only cache mid single-token fused decode (the
/// speculative-decode partial-accept rollback) must NOT discard the ring-only
/// tail: a subsequent decode step + `dequant` succeed with the correct full
/// prefix, not an abort or a zero-padded gap.
///
/// After prefill + `steps` fused decode steps the store carries a ring-only
/// tail. `truncate_to(prefill + keep)` rolls back the rejected tail; the GPU
/// ring is kept (mirroring the flat GPU-buffer codecs), so the kept prefix
/// `[0, prefill + keep)` survives in the ring. One more decode step then
/// overwrites position `prefill + keep`, and `dequant` returns the full
/// `prefill + keep + 1` prefix — byte/cosine-exact vs a CPU re-encode of the
/// surviving tokens.
///
/// Mutation check: re-introduce `self.gpu.clear()` in `QuantRotorK{3,4}::
/// truncate_to`. The ring (the only copy of the tail) is then dropped; the next
/// fused append `seed_from_cpu`s the frozen prefill blocks against the larger
/// `shape[2]` and errors (length mismatch), so `.expect("decode after
/// truncate")` panics — RED.
#[allow(clippy::expect_used, reason = "test: invariants documented")]
fn ring_only_tail_truncate_then_decode(quant: KvQuant) {
    let device = Device::Gpu;
    let (kv_h, n_q_heads, head_dim, prefill, steps, keep) =
        (2_i32, 8_i32, 128_i32, 6_i32, 5_i32, 2_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let pf_n = (prefill * kv_h * head_dim) as usize;
    let k_prefill = lcg_data(pf_n, 101);
    let step_ks: Vec<Vec<f32>> = (0..steps)
        .map(|s| lcg_data((kv_h * head_dim) as usize, 201 + s as u64))
        .collect();
    let k_new = lcg_data((kv_h * head_dim) as usize, 999);

    let mut cache = seeded_cache(quant, kv_h, head_dim, false);
    let k = f32_array(&k_prefill, &[1, kv_h, prefill, head_dim]);
    let v = f32_array(&lcg_data(pf_n, 102), &[1, kv_h, prefill, head_dim]);
    let q = f32_array(
        &lcg_data((prefill * n_q_heads * head_dim) as usize, 103),
        &[1, n_q_heads, prefill, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");
    for (i, ks_step) in step_ks.iter().enumerate() {
        let k1 = f32_array(ks_step, &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(
            &lcg_data((kv_h * head_dim) as usize, 301 + i as u64),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 401 + i as u64),
            &[1, n_q_heads, 1, head_dim],
        );
        cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa")
            .eval()
            .expect("decode out eval");
    }

    // Roll back to prefill + keep (partial-accept), then decode one more token.
    let m = prefill + keep;
    cache
        .truncate_to(m)
        .expect("a full-attention store rolls back to any prefix");
    assert_eq!(
        cache.offset(),
        m,
        "offset must roll back to the truncate target"
    );
    {
        let k1 = f32_array(&k_new, &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(
            &lcg_data((kv_h * head_dim) as usize, 777),
            &[1, kv_h, 1, head_dim],
        );
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 888),
            &[1, n_q_heads, 1, head_dim],
        );
        cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode after truncate")
            .eval()
            .expect("post-truncate decode eval");
    }

    let (dq, shape, _blocks_tokens, ring_live) = rotor_k_store_probe(&cache);
    assert!(
        ring_live,
        "ring must stay live across truncate (kept, not cleared)"
    );
    let final_seq = m + 1;
    assert_eq!(
        shape.get(2).copied().unwrap_or(0),
        final_seq,
        "shape[2] must be prefill+keep+1 after the post-truncate decode"
    );
    assert_eq!(
        dq.len(),
        (kv_h * final_seq * head_dim) as usize,
        "dequant must cover the full post-truncate prefix (no abort, no zero-pad)"
    );

    // Reference: prefill + the kept decode tokens + the new token.
    let mut surviving: Vec<Vec<f32>> = step_ks[..keep as usize].to_vec();
    surviving.push(k_new);
    let ref_dq = cpu_reference_dequant(quant, kv_h, head_dim, &k_prefill, &surviving, prefill);
    assert_eq!(ref_dq.len(), dq.len(), "reference length mismatch");
    let cos = mean_cosine(&dq, &ref_dq);
    assert!(
        cos > 0.99,
        "post-truncate dequant vs CPU reference cosine {cos} too low — the ring-only \
         tail was discarded (rejected/zeroed tokens) instead of the kept prefix"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_3_ring_only_tail_truncate_then_decode() {
    if skip_if_no_gpu_env() {
        return;
    }
    ring_only_tail_truncate_then_decode(KvQuant::RotorKOnly3);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_4_ring_only_tail_truncate_then_decode() {
    if skip_if_no_gpu_env() {
        return;
    }
    ring_only_tail_truncate_then_decode(KvQuant::RotorKOnly4);
}

/// A multi-token append AFTER fused decode has run lands on the block path, and
/// that path pushes a CPU block while keeping the ring.
///
/// This is the observable consequence `ring_feed_routing_tests` cannot reach:
/// those tests assert how `LEGACY_ROTOR_K_ONLY_FEED` routes, not that
/// `update_rotor_k_only_*` still passes it. Here the fused steps empty
/// `blocks` (the ring becomes the sole copy), and the following `q_seq > 1`
/// forward — a speculative verify chunk, or a continuation turn's prompt tokens
/// against a warm cache — falls out of the fused gate (`q_seq == 1`) into the
/// legacy entry. If that entry stopped passing `Maintain`, one of the two
/// assertions below flips: `MaintainRingOnly` would push no block, `Skip` would
/// drop the ring.
///
/// It also pins the reachability claim itself. The block arm was documented as
/// unreachable at `b == 1`; every shape here has `b == 1`.
#[allow(clippy::expect_used, reason = "test: invariants documented")]
fn multi_token_append_after_fused_decode_takes_the_block_path(quant: KvQuant) {
    let device = Device::Gpu;
    // `prefill = 24` matches every other dispatch test in this file; the value
    // is not load-bearing here but a lone outlier invites the question.
    let (kv_h, n_q_heads, head_dim, prefill, steps, second) =
        (2_i32, 8_i32, 128_i32, 24_i32, 4_i32, 3_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut cache = seeded_cache(quant, kv_h, head_dim, false);
    let pf_n = (prefill * kv_h * head_dim) as usize;
    let k = f32_array(&lcg_data(pf_n, 701), &[1, kv_h, prefill, head_dim]);
    let v = f32_array(&lcg_data(pf_n, 702), &[1, kv_h, prefill, head_dim]);
    let q = f32_array(
        &lcg_data((prefill * n_q_heads * head_dim) as usize, 703),
        &[1, n_q_heads, prefill, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    // Fused decode steps: these drop the CPU blocks once the ring is live.
    // Count the dispatches so a fused-arm fall-through is reported as such —
    // otherwise the `blocks_len == 0` precondition below fails and blames
    // `drop_blocks_when_ring_live_*` for a fault in the SDPA dispatcher.
    let before = rotor_flash_decode_dispatch_count();
    for step in 0..steps as u64 {
        let one = (kv_h * head_dim) as usize;
        let k1 = f32_array(&lcg_data(one, 710 + step), &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(&lcg_data(one, 720 + step), &[1, kv_h, 1, head_dim]);
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 730 + step),
            &[1, n_q_heads, 1, head_dim],
        );
        cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa")
            .eval()
            .expect("decode out eval");
    }
    let delta = rotor_flash_decode_dispatch_count() - before;
    assert!(
        delta >= steps as u64,
        "precondition: the fused flash kernel must have run for every decode step \
         ({delta} dispatches for {steps} steps). A shortfall is a dispatcher fault, \
         not a truncation one — read sdpa.rs before this file"
    );
    assert_eq!(
        rotor_k_blocks_len(&cache),
        0,
        "precondition: the fused decode path drops the CPU blocks once the ring is live"
    );

    // The transition under test: q_seq > 1 on the same cache, b == 1.
    let sec_n = (second * kv_h * head_dim) as usize;
    let k2 = f32_array(&lcg_data(sec_n, 740), &[1, kv_h, second, head_dim]);
    let v2 = f32_array(&lcg_data(sec_n, 741), &[1, kv_h, second, head_dim]);
    let q2 = f32_array(
        &lcg_data((second * n_q_heads * head_dim) as usize, 742),
        &[1, n_q_heads, second, head_dim],
    );
    cache
        .update_and_sdpa(&q2, &k2, &v2, scale, "causal", None, device)
        .expect("multi-token append after fused decode")
        .eval()
        .expect("out eval");

    assert!(
        rotor_k_blocks_len(&cache) > 0,
        "the legacy K-only entry must push a CPU block — a ring-only feed here would \
         leave `blocks` empty and the block arm would be dead after all"
    );
    assert!(
        rotor_k_ring_live(&cache),
        "the legacy K-only entry must keep the ring — a Skip feed here would strand the \
         prefix in the CPU blocks alone"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_3_multi_token_append_after_fused_decode_takes_the_block_path() {
    if skip_if_no_gpu_env() {
        return;
    }
    multi_token_append_after_fused_decode_takes_the_block_path(KvQuant::RotorKOnly3);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_k_only_4_multi_token_append_after_fused_decode_takes_the_block_path() {
    if skip_if_no_gpu_env() {
        return;
    }
    multi_token_append_after_fused_decode_takes_the_block_path(KvQuant::RotorKOnly4);
}

/// Sym sibling of [`multi_token_append_after_fused_decode_takes_the_block_path`],
/// pinning the **other** legacy feed.
///
/// `LEGACY_ROTOR_SYM_FEED` is `Skip`, so the block-path append drops the ring
/// and leaves the CPU blocks as the only copy of the prefix. That is the state
/// in which a mid-block speculative truncation is unrecoverable, which makes
/// this the feed the truncation fix matters most for — and, before this test,
/// the one with no behavioural coverage at all.
///
/// The three feeds are mutually exclusive **given the preconditions asserted
/// below**: `Skip` drops the ring and pushes a block, `Maintain` keeps the ring
/// and pushes a block (at `b == 1` the K-side ring feeder — `rotor3_sync_ring`
/// for `Rotor3Sym`, `rotor4_sync_ring` for `Rotor4Sym` — always ends with a
/// live ring via `gpu_append`), `MaintainRingOnly` pushes no block. Without the
/// preconditions the pair is vacuous — if the fused sym decode never dispatched,
/// the legacy append ran on every step, so `blocks_len > 0` is satisfied by the
/// prefill blocks alone and `!ring_live` by a ring that was never allocated, and
/// neither half of the transition was observed.
#[allow(clippy::expect_used, reason = "test: invariants documented")]
fn sym_multi_token_append_after_fused_decode_drops_the_ring(quant: KvQuant) {
    let device = Device::Gpu;
    let (kv_h, n_q_heads, head_dim, prefill, steps, second) =
        (2_i32, 8_i32, 128_i32, 24_i32, 4_i32, 3_i32);
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut cache = seeded_sym_cache(quant, kv_h, head_dim);
    let pf_n = (prefill * kv_h * head_dim) as usize;
    let k = f32_array(&lcg_data(pf_n, 801), &[1, kv_h, prefill, head_dim]);
    let v = f32_array(&lcg_data(pf_n, 802), &[1, kv_h, prefill, head_dim]);
    let q = f32_array(
        &lcg_data((prefill * n_q_heads * head_dim) as usize, 803),
        &[1, n_q_heads, prefill, head_dim],
    );
    cache
        .update_and_sdpa(&q, &k, &v, scale, "causal", None, device)
        .expect("prefill update_and_sdpa");
    cache.exit_prefill(device).expect("exit_prefill");

    // Count the symv dispatches so a fused-arm fall-through is reported as such
    // rather than silently satisfying both assertions below.
    let before = rotor_symv_flash_decode_dispatch_count();
    for step in 0..steps as u64 {
        let one = (kv_h * head_dim) as usize;
        let k1 = f32_array(&lcg_data(one, 810 + step), &[1, kv_h, 1, head_dim]);
        let v1 = f32_array(&lcg_data(one, 820 + step), &[1, kv_h, 1, head_dim]);
        let q1 = f32_array(
            &lcg_data((n_q_heads * head_dim) as usize, 830 + step),
            &[1, n_q_heads, 1, head_dim],
        );
        cache
            .update_and_sdpa(&q1, &k1, &v1, scale, "", None, device)
            .expect("decode update_and_sdpa")
            .eval()
            .expect("decode out eval");
    }

    // Preconditions. Without these the assertions after the transition pass
    // whether or not the fused path ever ran.
    let delta = rotor_symv_flash_decode_dispatch_count() - before;
    assert!(
        delta >= steps as u64,
        "precondition: the fused symv kernel must have run for every decode step \
         ({delta} dispatches for {steps} steps). A shortfall is a dispatcher fault, not \
         a feed one — read sdpa.rs before this file"
    );
    let (pre_blocks, pre_ring) = sym_k_store_state(&cache);
    assert_eq!(
        pre_blocks, 0,
        "precondition: the fused decode path drops the K-side CPU blocks once the ring \
         is live"
    );
    assert!(
        pre_ring,
        "precondition: the fused decode path leaves the K-side ring live — it is the \
         sole copy of the prefix going into the transition"
    );

    // The transition under test: q_seq > 1 on the same cache, b == 1.
    let sec_n = (second * kv_h * head_dim) as usize;
    let k2 = f32_array(&lcg_data(sec_n, 840), &[1, kv_h, second, head_dim]);
    let v2 = f32_array(&lcg_data(sec_n, 841), &[1, kv_h, second, head_dim]);
    let q2 = f32_array(
        &lcg_data((second * n_q_heads * head_dim) as usize, 842),
        &[1, n_q_heads, second, head_dim],
    );
    cache
        .update_and_sdpa(&q2, &k2, &v2, scale, "causal", None, device)
        .expect("multi-token append after fused decode")
        .eval()
        .expect("out eval");

    let (blocks_len, ring_live) = sym_k_store_state(&cache);
    assert!(
        blocks_len > 0,
        "the legacy sym entry must push a CPU block — a ring-only feed would leave \
         `blocks` empty and the store with no copy at all once the ring is dropped"
    );
    assert!(
        !ring_live,
        "the legacy sym entry must drop the ring — if it kept one, the CPU blocks \
         would no longer be the sole copy and the mid-block truncation hazard this \
         feed defines would not apply"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_sym3_multi_token_append_after_fused_decode_drops_the_ring() {
    if skip_if_no_gpu_env() {
        return;
    }
    sym_multi_token_append_after_fused_decode_drops_the_ring(KvQuant::Rotor3Sym);
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test rotor_flash_dispatch -- --ignored --test-threads=1"]
fn rotor_sym4_multi_token_append_after_fused_decode_drops_the_ring() {
    if skip_if_no_gpu_env() {
        return;
    }
    sym_multi_token_append_after_fused_decode_drops_the_ring(KvQuant::Rotor4Sym);
}
