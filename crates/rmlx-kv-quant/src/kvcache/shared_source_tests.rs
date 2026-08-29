//! CPU-device behaviour pins for `update_and_sdpa_shared_source`.
//!
//! These tests run on `Device::Cpu` and pin two structural properties of the
//! `update_and_sdpa_shared_source` code path that hold independently of GPU
//! availability:
//!
//! * **CPU-device dispatch gating**: with `Device::Cpu`, both
//!   `sdpa_dispatch_no_lock` (TurboFlash) and `try_fused_qk_dispatch`
//!   (fused-QK) early-return `Ok(None)` — `device != Device::Gpu` is
//!   an explicit pre-condition of both dispatch arms. This is a valuable
//!   property to pin: it proves the fallback gate is unconditional on CPU,
//!   regardless of env-var state.
//!
//! * **Legacy-path shape contract**: the legacy bf16 SDPA fallback (the path
//!   that runs when all dispatch arms return `None`) must surface K/V shaped
//!   `[B, kv_h, offset, D]`. Consumer layers (Gemma4 shared-KV) depend on
//!   this shape; a regression here would corrupt their attention computation.
//!
//! **What these tests do NOT cover**: GPU kernel dispatch (delta > 0).
//! That is covered by the integration test at
//! `crates/rmlx-kv-quant/tests/shared_source_dispatch.rs` which runs
//! under `#[ignore]` and requires `--test-threads=1` and a Metal GPU.

use super::core::KvCache;
use super::SharedKv;
use crate::kvcache::fused_qk_total_dispatch_count;
use crate::storage::KvStorage;
use crate::turbo_flash_msl::turbo_flash_dispatch_count;
use crate::KvQuant;
use rmlx_mlx::{Array, Device, Dtype};

fn f32_arr(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: Apple-Silicon-only build (CLAUDE.md Hard rule 1); f32 is 4-byte
    // LE on this target. `data` is borrowed read-only and the bytes are
    // copied into MLX before the borrow ends.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

const TEST_KV_H: i32 = 8;
const TEST_HEAD_DIM: i32 = 128;
const TEST_MAX_SEQ: i32 = 512;
const TEST_PREFILL_SEQ: i32 = 64;

/// CPU-device gate: `Device::Cpu` suppresses all dispatch arms regardless of
/// env-var state.
///
/// Both `sdpa_dispatch_no_lock` and `try_fused_qk_dispatch` return
/// `Ok(None)` when `device != Device::Gpu`. This test drives a K8V4 cache
/// through a prefill chunk + 1 decode step on `Device::Cpu` and asserts
/// that neither the TurboFlash counter nor the fused-QK counter moved.
///
/// This is the structural CPU-gate pin — it proves that the legacy fallback
/// is unconditional on CPU hosts regardless of `RMLX_TURBO_FLASH` or
/// `RMLX_FUSED_QK` env-var state. It does NOT prove that GPU dispatch works.
#[test]
fn shared_source_cpu_device_suppresses_all_dispatch_arms() {
    // Defensive: clear gates if a prior process / test latched them. These
    // gates use OnceLock so the value is per-process; the test relies on the
    // canonical default-off state.
    // SAFETY: env var mutation is single-threaded here (each #[test] runs
    // on its own thread; we do not spawn threads in this test).
    // We don't *set* anything; rely on default unset.
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V4, TEST_MAX_SEQ);

    cache.enter_prefill();

    let prefill_shape = [1_i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.111f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.222f32; n_pref], &prefill_shape);
    let q_pref_shape = [1_i32, 16, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_q_pref: usize = q_pref_shape.iter().map(|&d| d as usize).product();
    let q_pref = f32_arr(&vec![0.333f32; n_q_pref], &q_pref_shape);

    // Prefill via the shared-source entry — mirrors Gemma4's call site.
    let tf_before = turbo_flash_dispatch_count();
    let fqk_before = fused_qk_total_dispatch_count();
    let (_out, _share) = cache
        .update_and_sdpa_shared_source(&q_pref, &k_pref, &v_pref, 1.0, "causal", None, device)
        .expect("prefill shared source");
    cache.exit_prefill(device).expect("exit_prefill");

    // Decode step 1 — q_seq = 1.
    let step_kv_shape = [1_i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_kv_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.444f32; n_step], &step_kv_shape);
    let v_step = f32_arr(&vec![0.555f32; n_step], &step_kv_shape);
    let q_step_shape = [1_i32, 16, 1, TEST_HEAD_DIM];
    let n_q_step: usize = q_step_shape.iter().map(|&d| d as usize).product();
    let q_step = f32_arr(&vec![0.666f32; n_q_step], &q_step_shape);

    let (_out2, _share2) = cache
        .update_and_sdpa_shared_source(&q_step, &k_step, &v_step, 1.0, "", None, device)
        .expect("decode shared source");

    let tf_after = turbo_flash_dispatch_count();
    let fqk_after = fused_qk_total_dispatch_count();

    assert_eq!(
        tf_after, tf_before,
        "CPU-gate: TurboFlash counter advanced ({tf_before} -> {tf_after}) on \
         Device::Cpu — sdpa_dispatch_no_lock must gate off unconditionally on CPU."
    );
    assert_eq!(
        fqk_after, fqk_before,
        "CPU-gate: fused-QK counter advanced ({fqk_before} -> {fqk_after}) on \
         Device::Cpu — try_fused_qk_dispatch must gate off unconditionally on CPU."
    );
}

/// CPU-device legacy shape contract: surfaced K/V is shaped `[B, kv_h, offset, D]`.
///
/// The shared-KV consumer (Gemma4 cross-layer-KV) depends on `(K, V)` being
/// shaped `[B, kv_h, offset, D]` after `update_and_sdpa_shared_source`. This
/// test exercises a decode step on `Device::Cpu` (where all dispatch arms
/// gate off) and verifies the legacy bf16 fallback surfaces the full kv
/// prefix with the correct shape.
///
/// This pins the legacy-path shape contract. It does NOT exercise the GPU
/// dispatch path — shape correctness after TurboFlash dispatch is verified
/// in `crates/rmlx-kv-quant/tests/shared_source_dispatch.rs`.
#[test]
fn shared_source_cpu_device_legacy_fallback_surfaces_full_prefix_shape() {
    let device = Device::Cpu;
    let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V4, TEST_MAX_SEQ);

    cache.enter_prefill();
    let prefill_shape = [1_i32, TEST_KV_H, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.011f32; n_pref], &prefill_shape);
    let v_pref = f32_arr(&vec![0.022f32; n_pref], &prefill_shape);
    let q_pref_shape = [1_i32, 16, TEST_PREFILL_SEQ, TEST_HEAD_DIM];
    let n_q_pref: usize = q_pref_shape.iter().map(|&d| d as usize).product();
    let q_pref = f32_arr(&vec![0.033f32; n_q_pref], &q_pref_shape);
    let _ = cache
        .update_and_sdpa_shared_source(&q_pref, &k_pref, &v_pref, 1.0, "causal", None, device)
        .expect("prefill shared source");
    cache.exit_prefill(device).expect("exit_prefill");

    let step_kv_shape = [1_i32, TEST_KV_H, 1, TEST_HEAD_DIM];
    let n_step: usize = step_kv_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.044f32; n_step], &step_kv_shape);
    let v_step = f32_arr(&vec![0.055f32; n_step], &step_kv_shape);
    let q_step_shape = [1_i32, 16, 1, TEST_HEAD_DIM];
    let n_q_step: usize = q_step_shape.iter().map(|&d| d as usize).product();
    let q_step = f32_arr(&vec![0.066f32; n_q_step], &q_step_shape);

    let (_out, share) = cache
        .update_and_sdpa_shared_source(&q_step, &k_step, &v_step, 1.0, "", None, device)
        .expect("decode shared source");

    // Every dispatch arm gates off on CPU, so this is the legacy bf16 share.
    let SharedKv::Bf16(k_full, v_full) = &share else {
        panic!("the CPU legacy fallback must surface a bf16 share");
    };
    let expected = [1_i32, TEST_KV_H, TEST_PREFILL_SEQ + 1, TEST_HEAD_DIM];
    assert_eq!(
        k_full.shape(),
        expected,
        "surfaced K must span [0:offset] over the full prefix",
    );
    assert_eq!(
        v_full.shape(),
        expected,
        "surfaced V must span [0:offset] over the full prefix",
    );
}

/// A `Mixed` cache that was never declared a cross-layer-KV producer refuses
/// `update_and_sdpa_shared_source` instead of answering with a zeroed prefix.
///
/// The mirror this path surfaces is built at `exit_prefill`, gated on the
/// cache's own `shares_kv`. Reaching here without it means the model's topology
/// and its cache construction disagree; prefill is over, so the mirror cannot
/// be rebuilt — `update_decode_fp16` would allocate zeros and slice in only the
/// current token. The refusal happens before any offset moves, so the caller
/// sees an error rather than a silently wrong K/V.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn shared_source_refuses_a_mixed_cache_that_did_not_declare_sharing() {
    let device = Device::Cpu;
    let quant = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let mut cache = KvCache::with_quant_max_seq(quant, 512).with_shares_kv(false);
    // Bracket the same way the positive control below does, so the two arms
    // differ in `shares_kv` and in nothing else.
    cache.enter_prefill();

    let seq = 32_i32;
    let kv_h = 2_i32;
    let head_dim = 64_i32;
    let shape = [1_i32, kv_h, seq, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k = f32_arr(&vec![0.1_f32; n], &shape);
    let v = f32_arr(&vec![0.2_f32; n], &shape);
    let q = f32_arr(&vec![0.3_f32; n], &shape);

    let offset_before = cache.offset();
    let msg = match cache.update_and_sdpa_shared_source(&q, &k, &v, 1.0, "causal", None, device) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a Mixed cache with no declared sharing must refuse this path"),
    };
    assert!(
        msg.contains("with_shares_kv"),
        "the error must name the fix; got: {msg}"
    );
    assert_eq!(
        cache.offset(),
        offset_before,
        "the refusal must happen before any state moves"
    );

    // The same call on a cache that did declare sharing goes through.
    let mut shared = KvCache::with_quant_max_seq(quant, 512).with_shares_kv(true);
    shared.enter_prefill();
    shared
        .update_and_sdpa_shared_source(&q, &k, &v, 1.0, "causal", None, device)
        .expect("a declared shared-KV producer must be served");
}

/// A `Mixed` cache that **declared** cross-layer sharing but was rebuilt from
/// its packed store alone is refused too — the declaration is not the mirror.
///
/// This is the SSD-hydrate shape, reproduced through the constructor the SSD
/// reader itself calls: [`KvCache::from_storage`] restores the packed payload
/// and the block's `seq_len` as the offset, threads the arch's `shares_kv`
/// through, and leaves `decode_fp16_{k,v}` empty — the block carries no bf16
/// mirror for a `Mixed` layer, because that spill path only persists a mirror
/// for storages that hold no packed payload at all.
///
/// On the one architecture that shares K/V, such a block is tail-extended
/// through the prefix branch, which appends in **decode** mode with no
/// enter/exit prefill bracket. So `exit_prefill` never re-runs and the mirror
/// is never rebuilt. A guard that tests only the declaration lets this cache
/// straight into the zero-fill it exists to prevent: `update_decode_fp16` finds
/// no buffer, allocates `zeros`, and slices in the current token alone, so the
/// consumer layers attend a prefix of zeros with no error anywhere.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn shared_source_refuses_a_declared_producer_rebuilt_from_the_store_alone() {
    let device = Device::Cpu;
    let quant = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let kv_h = 2_i32;
    let head_dim = 64_i32;
    let prefix = 32_i32;

    // A real producer: declared, prefilled, mirror built by `exit_prefill`.
    let mut donor = KvCache::with_quant_max_seq(quant, 512).with_shares_kv(true);
    donor.enter_prefill();
    let shape = [1_i32, kv_h, prefix, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    donor
        .update(
            &f32_arr(&vec![0.125_f32; n], &shape),
            &f32_arr(&vec![0.25_f32; n], &shape),
            device,
        )
        .expect("prefill chunk");
    donor.exit_prefill(device).expect("exit_prefill");
    assert!(
        donor.decode_fp16_k_for_test().is_some(),
        "precondition: a declared producer holds its mirror after prefill"
    );

    // Spill + hydrate, structurally: the store survives, the mirror does not.
    let offset = donor.offset();
    let storage = std::mem::replace(&mut donor.storage, KvStorage::None { max_seq: 0 });
    let mut hydrated = KvCache::from_storage(
        storage,
        quant,
        offset,
        donor.layer_idx(),
        donor.dispatch_policy(),
        true,
    );
    assert!(
        hydrated.shares_kv() && hydrated.decode_fp16_k_for_test().is_none() && offset > 0,
        "precondition: the hydrated shape is a declared producer, past prefill, with no mirror"
    );

    let step = [1_i32, kv_h, 1_i32, head_dim];
    let m: usize = step.iter().map(|&d| d as usize).product();
    let q1 = f32_arr(&vec![0.5_f32; m], &step);
    let k1 = f32_arr(&vec![0.125_f32; m], &step);
    let v1 = f32_arr(&vec![0.25_f32; m], &step);

    let offset_before = hydrated.offset();
    match hydrated.update_and_sdpa_shared_source(&q1, &k1, &v1, 1.0, "", None, device) {
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains("SSD hydrate") && msg.contains("zero"),
                "the error must name the cause and the consequence; got: {msg}"
            );
            assert_eq!(
                hydrated.offset(),
                offset_before,
                "the refusal must happen before any state moves"
            );
        }
        Ok((_, SharedKv::Bf16(k_full, _))) => {
            // `Array::to_bytes` reads by raw linear offset, and the surfaced
            // mirror is a strided slice over the full `max_seq` buffer, so
            // flatten the view first — otherwise the count describes the
            // parent allocation rather than the share.
            let flat = k_full
                .contiguous(device)
                .expect("flatten the surfaced mirror");
            flat.eval().expect("materialise the surfaced mirror");
            let bytes = flat.to_bytes().expect("surfaced mirror bytes");
            let nonzero = bytes.iter().filter(|&&b| b != 0).count();
            let total = bytes.len();
            panic!(
                "the hydrated producer was served instead of refused: it surfaced a \
                 {:?} mirror of {total} bytes of which only {nonzero} are non-zero — \
                 every one of the {prefix} prefix tokens is the zero-fill this path \
                 must never answer with",
                flat.shape(),
            );
        }
        Ok((_, SharedKv::Store { kv_len })) => {
            panic!("the Mixed path must surface a bf16 share, got a store share of {kv_len} tokens")
        }
    }
}

/// A `Mixed` cache keeps its accepted prefix across `truncate_to`.
///
/// `truncate_to` is what a speculative round-loop calls to drop the rejected
/// tail of a draft block, and what the prompt cache calls to trim a slot. It
/// must leave the cache holding exactly the `n` positions the caller kept.
///
/// The Mixed store used to answer it by dropping the whole quant state, while
/// `KvCache::truncate_to` went on to set `offset = n`. The cache then reported
/// `n` positions and held none, which surfaces two different ways:
///
/// * the next **multi-token** forward (a speculative verify block) attends
///   `seq` keys against a mask sized from the reported offset — the opaque
///   `add: [broadcast_shapes] (1,kv,rep,seq,seq) and (1,1,seq,n+seq)` crash;
/// * the next **single-token** decode needs no mask, so it silently attends
///   the current token alone with no error anywhere.
///
/// The assertion has to reach the **quant store**, not the surfaced share: the
/// bf16 mirror this path hands to consumer layers is rebuilt from
/// `KvCache::offset`, so it spans the kept prefix either way and cannot tell
/// the two behaviours apart. An additive mask sized to the kept prefix does
/// reach it — that mask is added to the scores the store's own keys produce,
/// which is exactly the production failure.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn mixed_truncate_to_keeps_the_prefix_it_was_told_to_keep() {
    let device = Device::Cpu;
    let quant = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let kv_h = 2_i32;
    let head_dim = 64_i32;
    let prefix = 32_i32;
    let keep = 27_i32;

    let mut cache = KvCache::with_quant_max_seq(quant, 512).with_shares_kv(true);
    cache.enter_prefill();
    let shape = [1_i32, kv_h, prefix, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let k_pref = f32_arr(&vec![0.125_f32; n], &shape);
    let v_pref = f32_arr(&vec![0.25_f32; n], &shape);
    let q_pref = f32_arr(&vec![0.5_f32; n], &shape);
    cache
        .update_and_sdpa_shared_source(&q_pref, &k_pref, &v_pref, 1.0, "causal", None, device)
        .expect("prefill shared source");
    cache.exit_prefill(device).expect("exit_prefill");
    assert_eq!(cache.offset(), prefix, "precondition: prefix is resident");

    // Roll back a rejected draft tail.
    cache.truncate_to(keep);
    assert_eq!(
        cache.offset(),
        keep,
        "precondition: the cache reports the kept prefix"
    );

    // One more token, scored against a mask sized the way the caller sizes it:
    // from the offset the cache reports. The store must hold `keep + 1` keys
    // for that mask to apply.
    let step_shape = [1_i32, kv_h, 1, head_dim];
    let n_step: usize = step_shape.iter().map(|&d| d as usize).product();
    let k_step = f32_arr(&vec![0.75_f32; n_step], &step_shape);
    let v_step = f32_arr(&vec![0.875_f32; n_step], &step_shape);
    let q_step = f32_arr(&vec![0.375_f32; n_step], &step_shape);
    let mask_shape = [1_i32, 1, 1, keep + 1];
    let n_mask: usize = mask_shape.iter().map(|&d| d as usize).product();
    let mask = f32_arr(&vec![0.0_f32; n_mask], &mask_shape);

    let (out, share) = cache
        .update_and_sdpa_shared_source(&q_step, &k_step, &v_step, 1.0, "array", Some(&mask), device)
        .expect(
            "decode after truncate must attend the kept prefix — a broadcast error here means \
             the truncate dropped the store the mask was sized for",
        );
    out.eval().expect("eval decode output");

    assert_eq!(
        cache.offset(),
        keep + 1,
        "the append must land on top of the kept prefix"
    );
    assert_eq!(
        share.kv_len().expect("share kv_len"),
        keep + 1,
        "the surfaced K/V must span the kept prefix plus the new token"
    );
}
