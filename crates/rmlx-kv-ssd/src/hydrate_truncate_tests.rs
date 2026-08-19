//! Post-hydrate truncate round trips.
//!
//! Child of `hydrate_tests` so it can reuse that module's fixtures (`arr`,
//! `lcg`, `hash_to_hex_local`, the `MODEL_ID` / `TEST_*` constants) through
//! `super::*` rather than duplicating them.

use super::*;

// ── Post-hydrate truncate: the path where the CPU payload is load-bearing ─────
//
// On a normal serve these codecs never read their CPU block list at decode time:
// `exit_prefill` materialises the bf16 `decode_fp16_{k,v}` seed for every quant
// whose `feeds_bf16_k_at_decode()` is true, and every quantized `update_<codec>`
// early-returns into `update_decode_fp16` from then on — which is why
// `exit_prefill` does not build the store for them at all
// (`KvQuant::materialises_packed_store`). The fixtures below drive the codec
// body directly, without a prefill bracket, so the store exists to be cut.
//
// A hydrated cache is the exception, and it is what makes the block cut
// observable. `KvCache::from_storage` leaves `decode_fp16_k: None`, so the codec
// arm runs on every decode step — the store's blocks ARE the cache. A
// speculative rollback or a prompt-cache partial-prefix trim on such a cache
// lands mid-block (the hydrated prefix arrives as one block), and without the
// cut the next `update` stacks on top of the tokens the trim discarded.
//
// The oracles share no arithmetic with the truncation logic: the retained prefix
// is compared against a decode of the SAME store taken before the cut, and the
// appended correction against its own raw f32 source.

/// Build a single-layer cache of `quant` with `kv_h` heads and `seq` tokens.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn build_kvcache_quant(quant: KvQuant, kv_h: i32, seq: i32, seed: u64) -> KvCache {
    let device = Device::Cpu;
    let mut c = KvCache::with_quant_max_seq(quant, 4096);
    let shape = [1i32, kv_h, seq, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = arr(&lcg(n, seed), &shape);
    let v = arr(&lcg(n, seed ^ 0xABCD), &shape);
    // Deliberately NOT bracketed by `enter_prefill`/`exit_prefill`: a prefilled
    // cache of these codecs decodes off the bf16 mirror, so `exit_prefill`
    // builds no packed store and there would be nothing to spill as blocks.
    // The unbracketed append drives the codec body directly, which is the state
    // a hydrated cache is in and the one these fixtures exist to exercise.
    c.update(&k, &v, device).unwrap();
    c
}

/// Dequant the V side of a cache to flat head-major `[1, kv_h, S, D]` f32.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: every quant this file drives has a CPU-dequantizable V store; a None return is a bug in the test, not a runtime condition"
)]
fn probe_v(c: &KvCache, device: Device) -> Vec<f32> {
    // The two failure modes are kept apart: a missing V store is a bug in this
    // fixture, a refusing dequant is the blocks-vs-`shape[2]` coverage check
    // firing — which is the failure a broken cut produces, so it must not be
    // reported as "no V buffer".
    c.probe_v_dequant(device)
        .expect("probe_v: cache storage has no CPU-dequantizable V buffer")
        .expect("probe_v: V dequant refused — blocks do not cover shape[2]")
}

/// Spill a `seq`-token cache, hydrate it, truncate mid-block, append a
/// correction, and check the store holds the retained prefix plus the
/// correction — not the tokens the truncate discarded.
///
/// Mutation check: drop the `apply_truncate_plan` call from `QuantV::truncate_to`
/// (K8VTurbo3) or `QuantPlanarV::truncate_to` (Planar) and the correction
/// assertion fails on the discarded tokens' values. Before the coverage check
/// existed that failure was silent, which is the defect this pins.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: every index is derived from the shape the fixture was built with"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn hydrated_truncate_keeps_prefix_and_correction(quant: KvQuant, kv_h: i32) {
    let device = Device::Cpu;
    let head_dim = 128_i32;
    let prefill = BLOCK_TOKENS as i32; // 256 — arrives as one hydrated block
    let keep = 200_i32; // mid-block on purpose
    let corr = 2_i32;
    let label = format!("{quant} kv_h={kv_h}");

    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().to_path_buf();
    let index = SsdKvIndex::open_at(&dir.join("index.db")).unwrap();

    let prompt_ids: Vec<u32> = (0..prefill as u32).collect();
    let chained = chained_block_hashes_seeded(
        &prompt_ids,
        cache_seed(TEST_LAYOUT_KEY, quant, &[quant], TEST_MODEL_SIG),
    );
    let key = hash_to_hex_local(chained[0]);
    let path = dir.join(format!("{key}.kvb"));

    let cache = build_kvcache_quant(quant, kv_h, prefill, 0x7A1E);
    write_caches(&path, device, MODEL_ID, quant, &[cache], &[]).unwrap();
    let size = std::fs::metadata(&path).unwrap().len();
    index
        .record(
            &key,
            TEST_LAYOUT_KEY,
            &path,
            MODEL_ID,
            &quant.to_string(),
            size,
        )
        .unwrap();

    let hydrator = SsdHydrator::with_index(MODEL_ID, TEST_LAYOUT_KEY, device, dir, index);
    let mut block = hydrator
        .lookup(
            &prompt_ids,
            cache_seed(TEST_LAYOUT_KEY, quant, &[quant], TEST_MODEL_SIG),
            quant,
            DispatchPolicy::default(),
        )
        .unwrap()
        .expect("SSD hit expected");
    let cache = &mut block.kv_caches[0];

    // Sanity, NOT the anti-vacuity guard: a hydrated cache is expected to carry
    // no bf16 decode seed. This is a weaker predicate than the one the codec
    // early-return keys on (`decode_fp16_k.is_some()` alone) — `decode_fp16_kv()`
    // returns `None` when EITHER buffer is unset, so a K-set/V-unset cache would
    // satisfy it while the codec arm is bypassed.
    //
    // What actually makes this test non-vacuous is the decode-length assertion
    // after the append: on the seeded path `update_decode_fp16` never touches
    // the store, so the decode would be `keep` positions long, not `keep + corr`.
    assert!(
        cache.decode_fp16_kv().is_none(),
        "{label}: a hydrated cache is expected to carry no bf16 decode seed"
    );

    // Reference decodes of the untruncated store — the oracles for the prefix.
    // Both axes: `truncate_to` cuts K as well, and `QuantK::retain_cpu_prefix` is
    // the only element-granular cut in the fix and the only one carrying
    // refusals, so leaving it unasserted would let a regression in it pass.
    let full_k = probe_k(cache, device);
    let full_v = probe_v(cache, device);
    assert_eq!(
        full_v.len(),
        (kv_h * prefill * head_dim) as usize,
        "{label}: full V decode length"
    );

    cache.truncate_to(keep);

    // The correction: distinct raw f32 from its own LCG stream.
    let corr_shape = [1_i32, kv_h, corr, head_dim];
    let corr_n: usize = corr_shape.iter().map(|&x| x as usize).product();
    let corr_k_raw = lcg(corr_n, 0xC0FF);
    let corr_v_raw = lcg(corr_n, 0xC0FF ^ 0xABCD);
    cache
        .update(
            &arr(&corr_k_raw, &corr_shape),
            &arr(&corr_v_raw, &corr_shape),
            device,
        )
        .unwrap();

    let after_k = probe_k(cache, device);
    let after_v = probe_v(cache, device);
    let total = keep + corr;
    // The anti-vacuity assertion: on the seeded decode path the store is never
    // appended to, so this would be `keep` positions, not `keep + corr`.
    for (axis, got) in [("K", &after_k), ("V", &after_v)] {
        assert_eq!(
            got.len(),
            (kv_h * total * head_dim) as usize,
            "{label} {axis}: store must decode to exactly the retained prefix plus \
             the correction"
        );
    }

    // 1. The retained prefix is bit-identical to the same positions of the
    //    pre-truncate decode, on BOTH axes. Head-major layout, so index per
    //    (head, position) — it is not a flat prefix.
    for (axis, after, full) in [("K", &after_k, &full_k), ("V", &after_v, &full_v)] {
        for h in 0..kv_h {
            for s in 0..keep {
                for dd in 0..head_dim {
                    let got = after[((h * total + s) * head_dim + dd) as usize];
                    let want = full[((h * prefill + s) * head_dim + dd) as usize];
                    assert!(
                        (got - want).abs() < f32::EPSILON,
                        "{label} {axis}: retained prefix changed at h={h} s={s} \
                         d={dd}: {got} vs {want}"
                    );
                }
            }
        }
    }

    // 2. The tail holds the CORRECTION, not the discarded tokens. The oracle is
    //    the raw f32 the correction was built from — no store, no truncation
    //    arithmetic.
    for (axis, after, full, raw) in [
        ("K", &after_k, &full_k, &corr_k_raw),
        ("V", &after_v, &full_v, &corr_v_raw),
    ] {
        let mut corr_err = 0.0_f32;
        let mut stale_err = 0.0_f32;
        for h in 0..kv_h {
            for s in 0..corr {
                for dd in 0..head_dim {
                    let got = after[((h * total + keep + s) * head_dim + dd) as usize];
                    let want = raw[((h * corr + s) * head_dim + dd) as usize];
                    let discarded = full[((h * prefill + keep + s) * head_dim + dd) as usize];
                    corr_err = corr_err.max((got - want).abs());
                    stale_err = stale_err.max((got - discarded).abs());
                }
            }
        }
        assert!(
            corr_err < 0.2,
            "{label} {axis}: the appended correction is missing — the tail differs \
             from its own source by {corr_err} (a pre-fix store returns the \
             discarded tokens here)"
        );
        assert!(
            stale_err > corr_err,
            "{label} {axis}: premise — the discarded tokens must be distinguishable \
             from the correction, else this assertion cannot fail \
             (corr_err={corr_err}, stale_err={stale_err})"
        );
    }
}

/// `K8VTurbo3` — `QuantV` (`Vec<TurboBlocks>`, 3-bit) on V, `QuantK` on K.
#[test]
fn hydrated_k8vturbo3_truncate_keeps_prefix_and_correction_kv_h_1() {
    hydrated_truncate_keeps_prefix_and_correction(KvQuant::K8VTurbo3, 1);
}

#[test]
fn hydrated_k8vturbo3_truncate_keeps_prefix_and_correction_kv_h_2() {
    hydrated_truncate_keeps_prefix_and_correction(KvQuant::K8VTurbo3, 2);
}

/// `Planar` — `QuantPlanarV` (`Vec<PlanarBlocks>`) on V, `QuantK` on K.
#[test]
fn hydrated_planar_truncate_keeps_prefix_and_correction_kv_h_1() {
    hydrated_truncate_keeps_prefix_and_correction(KvQuant::Planar, 1);
}

#[test]
fn hydrated_planar_truncate_keeps_prefix_and_correction_kv_h_2() {
    hydrated_truncate_keeps_prefix_and_correction(KvQuant::Planar, 2);
}
