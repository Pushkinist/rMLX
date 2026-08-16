use super::*;
use crate::KvQuant;
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

/// Regression test for the AN-SSD-canary panic:
///
/// After SSD hydration, a SWA layer is stored with `KvStorage::None`
/// (the spiller records it as tag "none" since the rotating bf16 ring
/// cannot be spilled), but `KvCache::from_storage` sets `self.quant` to
/// the model's global KvQuant (e.g. K8V8). The old `update()` dispatch
/// branched on `self.quant` → reached `update_k8v8()` → pattern-matched
/// `self.storage` expecting `KvStorage::K8V8` → `unreachable!()` panic.
///
/// The fix dispatches on `self.storage` discriminant instead. This test
/// exercises the exact mismatch path: `KvStorage::None` + `KvQuant::K8V8`
/// (the quant Gemma4 uses for full-attention layers). `update()` must
/// succeed and return a valid (k, v) pair.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn hydrate_none_storage_k8v8_quant_update_no_panic() {
    let device = Device::Cpu;
    // Simulate a hydrated SWA layer: from_storage sets quant=K8V8 but the
    // on-disk tag was "none" → storage=KvStorage::None.
    let mut cache = KvCache::from_storage(
        KvStorage::None { max_seq: 4096 },
        KvQuant::K8V8,
        256, // offset: 256 tokens already "cached"
        0,   // layer_idx: 0 for this test helper,
        DispatchPolicy::default(),
    );
    // One decode step: shape [B=1, kv_h=2, seq=1, D=128]
    let shape = [1i32, 2, 1, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);
    let result = cache.update(&k, &v, device);
    assert!(
        result.is_ok(),
        "update() on hydrated SWA layer (KvStorage::None + KvQuant::K8V8) must not panic: {result:?}"
    );
    let (k_out, v_out) = result.unwrap();
    // The output K and V must have a non-empty sequence dimension.
    assert!(k_out.shape()[2] > 0, "K output must have seq > 0");
    assert!(v_out.shape()[2] > 0, "V output must have seq > 0");
}

/// Regression guard for all non-K8V8 quants that a Gemma4 model might
/// spill: K8V4 and Planar also have a None-storage SWA layer after hydrate.
/// Exercise both so the dispatch stays correct if K8V8 is later fixed but
/// K8V4/Planar regress.
#[test]
fn hydrate_none_storage_k8v4_and_planar_quant_update_no_panic() {
    let device = Device::Cpu;
    let shape = [1i32, 2, 1, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);

    for quant in [KvQuant::K8V4, KvQuant::Planar] {
        let mut cache = KvCache::from_storage(
            KvStorage::None { max_seq: 4096 },
            quant,
            256,
            0,
            DispatchPolicy::default(),
        );
        let result = cache.update(&k, &v, device);
        assert!(
            result.is_ok(),
            "update() on hydrated SWA layer (KvStorage::None + {quant:?}) must not panic: {result:?}"
        );
    }
}

/// Regression test: `exit_prefill()` on a hydrated SWA layer
/// (`KvStorage::None` + `KvQuant::K8V8`) must not panic.
///
/// This is the second dispatch site that was broken by the quant/storage
/// mismatch. `exit_prefill()` at the `match self.quant` block matched
/// `KvQuant::K8V8` then pattern-matched `self.storage` expecting
/// `KvStorage::K8V8` → `unreachable!()` for hydrated SWA layers whose
/// storage is `KvStorage::None`. The guard added above the `match` block
/// takes the fp16 early-return path for any `KvStorage::None` cache.
#[test]
fn hydrate_none_storage_k8v8_quant_exit_prefill_no_panic() {
    let device = Device::Cpu;
    // Simulate a fresh hydrated SWA layer: offset=0, storage=None, quant=K8V8.
    let mut cache = KvCache::from_storage(
        KvStorage::None { max_seq: 4096 },
        KvQuant::K8V8,
        0,
        0,
        DispatchPolicy::default(),
    );
    cache.enter_prefill();
    // Feed a 4-token prefill chunk: shape [B=1, kv_h=2, seq=4, D=128]
    let shape = [1i32, 2, 4, 128];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);
    let upd = cache.update(&k, &v, device);
    assert!(
        upd.is_ok(),
        "update() during prefill on hydrated SWA layer must succeed: {upd:?}"
    );
    // exit_prefill must not panic
    let result = cache.exit_prefill(device);
    assert!(
        result.is_ok(),
        "exit_prefill() on hydrated SWA layer (KvStorage::None + KvQuant::K8V8) must not panic: {result:?}"
    );
}

/// KvCache built with KvQuant::Iso3 stores an IsoV3 variant.
#[test]
fn isov3_dispatch_routes_iso3_to_iso_v3() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Iso3, 128);
    assert!(
        matches!(cache.storage(), KvStorage::IsoV3 { .. }),
        "KvCache::with_quant(Iso3) should construct KvStorage::IsoV3"
    );
}

/// KvCache built with KvQuant::Iso4 stores an IsoV4 variant.
#[test]
fn isov4_dispatch_routes_iso4_to_iso_v4() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Iso4, 128);
    assert!(
        matches!(cache.storage(), KvStorage::IsoV4 { .. }),
        "KvCache::with_quant(Iso4) should construct KvStorage::IsoV4"
    );
}

/// KvCache built with KvQuant::Rotor3 stores a RotorV3 variant.
#[test]
fn rotorv3_dispatch_routes_rotor3_to_rotor_v3() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Rotor3, 128);
    assert!(
        matches!(cache.storage(), KvStorage::RotorV3 { .. }),
        "KvCache::with_quant(Rotor3) should construct KvStorage::RotorV3"
    );
}

/// KvCache::with_quant routes Iso3Sym to KvStorage::IsoSym3.
#[test]
fn iso_sym3_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Iso3Sym, 128);
    assert!(
        matches!(cache.storage(), KvStorage::IsoSym3 { .. }),
        "KvCache::with_quant(Iso3Sym) should construct KvStorage::IsoSym3"
    );
}

/// Iso4Sym → IsoSym4.
#[test]
fn iso_sym4_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Iso4Sym, 128);
    assert!(matches!(cache.storage(), KvStorage::IsoSym4 { .. }));
}

/// IsoKOnly3 → KvStorage::IsoKOnly3.
#[test]
fn iso_k_only_3_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly3, 128);
    assert!(matches!(cache.storage(), KvStorage::IsoKOnly3 { .. }));
}

/// IsoKOnly4 → KvStorage::IsoKOnly4.
#[test]
fn iso_k_only_4_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly4, 128);
    assert!(matches!(cache.storage(), KvStorage::IsoKOnly4 { .. }));
}

/// Regression: `update_iso_k_only_3` must NOT populate
/// `decode_fp16_k` as a side-effect of the V-side bf16 update.
///
/// Before the fix, `update_iso_k_only_3` called `update_decode_fp16` for
/// V, which populated `self.decode_fp16_k`. On the second decode step the
/// `decode_fp16_k.is_some()` guard fired the bf16 early-return path,
/// silently serving raw bf16 K instead of the ISO-quantized path.
///
/// After the fix the helper is `update_decode_fp16_v_only`, which never
/// touches `decode_fp16_k`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: update must succeed on a freshly constructed cache"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "shape indices established by construction"
)]
fn iso_k_only_3_decode_fp16_k_stays_none_after_update() {
    let device = Device::Cpu;
    let shape = [1i32, 2, 1, 64];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);

    let mut cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly3, 128);
    // First decode step.
    let r1 = cache.update(&k, &v, device);
    assert!(r1.is_ok(), "first update must succeed: {r1:?}");
    assert!(
        cache.decode_fp16_k_for_test().is_none(),
        "decode_fp16_k must remain None after first IsoKOnly3 update (step 1)"
    );

    // Second decode step — this was the failing case before the fix.
    let r2 = cache.update(&k, &v, device);
    assert!(r2.is_ok(), "second update must succeed: {r2:?}");
    assert!(
        cache.decode_fp16_k_for_test().is_none(),
        "decode_fp16_k must remain None after second IsoKOnly3 update (step 2)"
    );
}

/// Regression: same guard for IsoKOnly4.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: update must succeed on a freshly constructed cache"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "shape indices established by construction"
)]
fn iso_k_only_4_decode_fp16_k_stays_none_after_update() {
    let device = Device::Cpu;
    let shape = [1i32, 2, 1, 64];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);

    let mut cache = KvCache::with_quant_max_seq(KvQuant::IsoKOnly4, 128);
    // First decode step.
    let r1 = cache.update(&k, &v, device);
    assert!(r1.is_ok(), "first update must succeed: {r1:?}");
    assert!(
        cache.decode_fp16_k_for_test().is_none(),
        "decode_fp16_k must remain None after first IsoKOnly4 update (step 1)"
    );

    // Second decode step — this was the failing case before the fix.
    let r2 = cache.update(&k, &v, device);
    assert!(r2.is_ok(), "second update must succeed: {r2:?}");
    assert!(
        cache.decode_fp16_k_for_test().is_none(),
        "decode_fp16_k must remain None after second IsoKOnly4 update (step 2)"
    );
}

// ── Rotor dispatch + regression tests ──────────────────────────────────────

#[test]
fn rotor_sym3_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Rotor3Sym, 128);
    assert!(matches!(cache.storage(), KvStorage::RotorSym3 { .. }));
}

#[test]
fn rotor_sym4_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::Rotor4Sym, 128);
    assert!(matches!(cache.storage(), KvStorage::RotorSym4 { .. }));
}

#[test]
fn rotor_k_only_3_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::RotorKOnly3, 128);
    assert!(matches!(cache.storage(), KvStorage::RotorKOnly3 { .. }));
}

#[test]
fn rotor_k_only_4_dispatch() {
    let cache = KvCache::with_quant_max_seq(KvQuant::RotorKOnly4, 128);
    assert!(matches!(cache.storage(), KvStorage::RotorKOnly4 { .. }));
}

/// Regression: `update_rotor_k_only_3` must NOT populate `decode_fp16_k` as a side-effect.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: update must succeed on a freshly constructed cache"
)]
fn rotor_k_only_3_decode_fp16_k_stays_none_after_update() {
    let device = Device::Cpu;
    let shape = [1i32, 2, 1, 64];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);

    let mut cache = KvCache::with_quant_max_seq(KvQuant::RotorKOnly3, 128);
    let r1 = cache.update(&k, &v, device);
    assert!(r1.is_ok(), "first update must succeed: {r1:?}");
    assert!(
        cache.decode_fp16_k_for_test().is_none(),
        "decode_fp16_k must remain None after first RotorKOnly3 update (step 1)"
    );
    let r2 = cache.update(&k, &v, device);
    assert!(r2.is_ok(), "second update must succeed: {r2:?}");
    assert!(
        cache.decode_fp16_k_for_test().is_none(),
        "decode_fp16_k must remain None after second RotorKOnly3 update (step 2)"
    );
}

/// Regression: same guard for RotorKOnly4.
#[test]
fn rotor_k_only_4_decode_fp16_k_stays_none_after_update() {
    let device = Device::Cpu;
    let shape = [1i32, 2, 1, 64];
    let n: usize = shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n], &shape);
    let v = f32_arr(&vec![0.2f32; n], &shape);

    let mut cache = KvCache::with_quant_max_seq(KvQuant::RotorKOnly4, 128);
    let r1 = cache.update(&k, &v, device);
    assert!(r1.is_ok());
    assert!(cache.decode_fp16_k_for_test().is_none());
    let r2 = cache.update(&k, &v, device);
    assert!(r2.is_ok());
    assert!(cache.decode_fp16_k_for_test().is_none());
}

/// CRITICAL regression — `update_and_sdpa_planar_k_fused` MUST
/// pre-increment `self.offset` before calling `update_decode_fp16_v_only`.
///
/// Before the fix, the fused-QK helper called the bf16 V-update helper
/// without advancing `self.offset`.  The helper computes the write window as
/// `prev_offset = self.offset - new_seq` / `new_offset = self.offset`, so
/// every decode step wrote V at `[offset-1, offset)` — overwriting the LAST
/// written position — and `self.offset` never grew past `new_seq`.  Result:
/// V buffer permanently lagged by one token, accumulator silently degraded.
///
/// This test goes through `update_and_sdpa` and asserts the offset advances
/// monotonically per call.  GPU-gated because the fused path requires
/// `Device::Gpu` (CPU falls through to the legacy dequant+SDPA path).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_k_fused_offset -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: update must succeed on a freshly constructed cache"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "shape indices established by construction"
)]
fn planar_k_fused_offset_advances_per_decode_step() {
    use crate::test_utils::skip_if_no_gpu_env;
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    // Decode-step shapes: [B=1, kv_h=2, S=1, D=64].  Q has n_q_heads=2
    // (heads_per_kv=1, MHA-style — keeps the test minimal).
    let kv_shape = [1i32, 2, 1, 64];
    let q_shape = [1i32, 2, 1, 64];
    let n_kv: usize = kv_shape.iter().map(|&x| x as usize).product();
    let n_q: usize = q_shape.iter().map(|&x| x as usize).product();
    let k = f32_arr(&vec![0.1f32; n_kv], &kv_shape);
    let v_step1 = f32_arr(&vec![0.2f32; n_kv], &kv_shape);
    let v_step2 = f32_arr(&vec![0.7f32; n_kv], &kv_shape);
    let q = f32_arr(&vec![0.05f32; n_q], &q_shape);

    let mut cache = KvCache::with_quant_max_seq(KvQuant::PlanarK, 128);
    let initial_offset = cache.offset();

    // Step 1: cache should grow by 1.
    let r1 = cache.update_and_sdpa(&q, &k, &v_step1, 1.0, "", None, device);
    assert!(r1.is_ok(), "first update_and_sdpa must succeed: {r1:?}");
    assert_eq!(
        cache.offset(),
        initial_offset + 1,
        "offset must advance by 1 after first decode step"
    );

    // Step 2: cache should grow by 1 more (total +2).  Before the fix,
    // offset stayed at `initial_offset + 0` because update_decode_fp16
    // wrote at `[offset-1, offset)` and did NOT advance offset itself.
    let r2 = cache.update_and_sdpa(&q, &k, &v_step2, 1.0, "", None, device);
    assert!(r2.is_ok(), "second update_and_sdpa must succeed: {r2:?}");
    assert_eq!(
        cache.offset(),
        initial_offset + 2,
        "offset must advance by 2 after two decode steps (was permanently lagged before fix)"
    );
}
