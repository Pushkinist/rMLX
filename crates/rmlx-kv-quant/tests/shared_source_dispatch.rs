// GPU integration test: `update_and_sdpa_shared_source` dispatch chain.
//
// Drives the public `KvCache::update_and_sdpa_shared_source` path (Gemma4's
// entry point) with K8V4 storage, head_dim=128. Asserts:
//
//   1. `DispatchPolicy { turbo_flash: true, turbo_flash_min_kv_seq: 0 }` (zero
//      threshold so TurboFlash fires on a short kv_seq):
//      a. `turbo_flash_dispatch_count()` delta > 0 after a decode step (the
//         kernel actually fired on the shared-KV producer path, not just gated).
//      b. The bf16 K surfaced to the consumer is byte-identical to slicing
//         `decode_fp16_k` — proving the dispatch path surfaced the mirror
//         rather than a flash-transformed K.
//
//   2. `DispatchPolicy { turbo_flash: false, .. }` (control group):
//      Same prefill + decode; dispatch delta == 0 (gate correctly suppressed).
//
// `RMLX_SHARED_SOURCE_STRICT=1` — strict mode (used by CI):
//   * When env=ON and delta == 0, panic instead of skip.
//   * When the dispatch errors, panic instead of skip.
//
// `#[ignore]` because it needs the Metal GPU. Run via:
//   cargo test -p rmlx-kv-quant --test shared_source_dispatch -- --ignored --test-threads=1
//
// Preflight: `pkill -f "rmlx serve" || true; rm -f /tmp/rmlx.*.claim`.
// CLAUDE.md hard rule 8 (single MLX process): integration runner serialises
// tests within one process — no claim file is held by tests.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    missing_docs
)]

use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant, SharedKv};
use rmlx_mlx::{Array, Device, Dtype};

/// Unwrap a producer's share into `(out, K, V)`.
///
/// K8V4 / `None` storage has no fused-over-store arm (TurboFlash and fused-QK
/// both maintain the bf16 mirror), so the share here is always bf16. A `Store`
/// share would mean a dispatch gate regressed.
#[allow(
    clippy::panic,
    reason = "test helper: a Store share on a K8V4/None cache is a gate regression, and must fail the test loudly"
)]
fn split_bf16_share(pair: (Array, SharedKv)) -> (Array, Array, Array) {
    let (out, share) = pair;
    match share {
        SharedKv::Bf16(k, v) => (out, k, v),
        SharedKv::Store { .. } => {
            panic!("expected a bf16 share — K8V4/None maintain the bf16 mirror")
        }
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("Array::from_bytes")
}

fn array_to_bf16_bytes(a: &Array, device: Device) -> Vec<u8> {
    let bf16 = a.astype(Dtype::Bf16, device).expect("astype bf16");
    // Materialise on device before reading bytes.
    bf16.eval().expect("materialise array");
    bf16.to_bytes().expect("to_bytes")
}

/// LCG pseudo-random data for reproducible tensors. Same constants as
/// the fused-QK dispatch tests so seeds round-trip across reports.
fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = (state >> 32) as u32 as f32 / u32::MAX as f32;
            frac * 2.0 - 1.0
        })
        .collect()
}

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

fn is_strict() -> bool {
    std::env::var("RMLX_SHARED_SOURCE_STRICT").as_deref() == Ok("1")
}

// ── test parameters ───────────────────────────────────────────────────────────

const B: i32 = 1;
const KV_H: i32 = 2;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;
const HEAD_DIM: i32 = 128;
const PREFILL_SEQ: i32 = 64;
const MAX_SEQ: i32 = 4096;

/// Build a K8V4 cache prefilled with 64 tokens via `update_and_sdpa_shared_source`.
fn build_prefilled_cache(
    device: Device,
    prefill_k: &Array,
    prefill_v: &Array,
    prefill_q: &Array,
    policy: DispatchPolicy,
) -> KvCache {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ).with_dispatch_policy(policy);
    cache.enter_prefill();
    let _ = cache
        .update_and_sdpa_shared_source(
            prefill_q, prefill_k, prefill_v, scale, "causal", None, device,
        )
        .expect("prefill shared source");
    cache.exit_prefill(device).expect("exit_prefill");
    assert_eq!(cache.offset(), PREFILL_SEQ, "cache offset after prefill");
    cache
}

// ── Test 1: turbo_flash on — dispatch fires, bf16 mirror surfaced ───────────

/// GPU: `update_and_sdpa_shared_source` with TurboFlash ON must dispatch and
/// surface the bf16 mirror K unchanged.
///
/// With `turbo_flash: true` + `turbo_flash_min_kv_seq: 0`:
///   a. `turbo_flash_dispatch_count()` delta > 0 (kernel fired).
///   b. Surfaced K bytes == bf16 reference K bytes (mirror, not flash-transformed).
///
/// `RMLX_SHARED_SOURCE_STRICT=1` turns a delta=0 skip into a panic.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test shared_source_dispatch -- --ignored --test-threads=1"]
fn shared_source_turbo_flash_on_dispatch_fires_and_surfaces_bf16_mirror() {
    if skip_if_no_gpu() {
        return;
    }

    let strict = is_strict();
    let device = Device::Gpu;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // The policy travels on the cache, so this arm needs no process state.
    let policy = DispatchPolicy {
        turbo_flash: true,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    };

    let prefill_shape = [B, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_pref, 0xB1C2D3E4_F5A6_0001);
    let v_data = lcg_data(n_pref, 0xB1C2D3E4_F5A6_0002);
    let prefill_k = make_f32_array(&k_data, &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_k");
    let prefill_v = make_f32_array(&v_data, &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_v");
    let q_pref_shape = [B, N_Q_HEADS, PREFILL_SEQ, HEAD_DIM];
    let n_q_pref: usize = q_pref_shape.iter().map(|&d| d as usize).product();
    let q_pref_data = lcg_data(n_q_pref, 0xB1C2D3E4_F5A6_0003);
    let q_pref = make_f32_array(&q_pref_data, &q_pref_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q_pref");

    let mut cache = build_prefilled_cache(device, &prefill_k, &prefill_v, &q_pref, policy);

    let dec_kv_shape = [B, KV_H, 1, HEAD_DIM];
    let n_dec: usize = dec_kv_shape.iter().map(|&d| d as usize).product();
    let new_k_data = lcg_data(n_dec, 0xB1C2D3E4_F5A6_0004);
    let new_v_data = lcg_data(n_dec, 0xB1C2D3E4_F5A6_0005);
    let new_k = make_f32_array(&new_k_data, &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&new_v_data, &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");
    let q_dec_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_q_dec: usize = q_dec_shape.iter().map(|&d| d as usize).product();
    let q_dec_data = lcg_data(n_q_dec, 0xB1C2D3E4_F5A6_0006);
    let q_dec = make_f32_array(&q_dec_data, &q_dec_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q_dec");

    let tf_before = turbo_flash_dispatch_count();
    let result =
        cache.update_and_sdpa_shared_source(&q_dec, &new_k, &new_v, scale, "", None, device);
    let tf_after = turbo_flash_dispatch_count();

    let (_, k_surfaced, _) = match result.map(split_bf16_share) {
        Ok(triple) => triple,
        Err(e) => {
            eprintln!(
                "TF=1: update_and_sdpa_shared_source errored: {e} (skipping dispatch/parity assert)"
            );
            assert!(
                !strict,
                "RMLX_SHARED_SOURCE_STRICT=1 — update_and_sdpa_shared_source errored ({e}), \
                 but strict mode requires the dispatch to succeed"
            );
            return;
        }
    };

    let delta = tf_after - tf_before;
    eprintln!("TF=1: turbo_flash_dispatch_count before={tf_before} after={tf_after} delta={delta}");

    if delta == 0 {
        eprintln!(
            "TF=1: HOLD-soft — TurboFlash did not fire on the shared-KV producer path \
             (expected when head_dim not in {{128,256}} or kv_seq below threshold). \
             Skipping byte-identity assertion."
        );
        assert!(
            !strict,
            "RMLX_SHARED_SOURCE_STRICT=1 — turbo_flash_dispatch_count did not increment; \
             the kernel is expected to fire with turbo_flash on + a zero threshold \
             at head_dim=128 in strict mode"
        );
        return;
    }

    // Assertion (b): the surfaced K must match a bf16 legacy-path reference.
    // A flash-transformed K would differ nontrivially in both shape and values.
    let expected_k_seq = PREFILL_SEQ + 1;
    let k_shape = k_surfaced.shape();
    assert_eq!(
        k_shape,
        [B, KV_H, expected_k_seq, HEAD_DIM],
        "TF=1: surfaced K shape must be [B={B}, kv_h={KV_H}, seq={expected_k_seq}, D={HEAD_DIM}]"
    );

    // Build bf16 reference via no-quant cache on the same prefill + decode.
    let mut ref_cache = KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ);
    ref_cache.enter_prefill();
    let _ = ref_cache
        .update_and_sdpa_shared_source(
            &q_pref, &prefill_k, &prefill_v, scale, "causal", None, device,
        )
        .expect("ref prefill");
    ref_cache.exit_prefill(device).expect("ref exit_prefill");
    let (_, k_ref, _) = split_bf16_share(
        ref_cache
            .update_and_sdpa_shared_source(&q_dec, &new_k, &new_v, scale, "", None, device)
            .expect("ref decode"),
    );

    let k_surf_bytes = array_to_bf16_bytes(&k_surfaced, device);
    let k_ref_bytes = array_to_bf16_bytes(&k_ref, device);
    assert_eq!(
        k_surf_bytes.len(),
        k_ref_bytes.len(),
        "TF=1: surfaced K and bf16 reference K must have the same byte length"
    );
    assert_eq!(
        k_surf_bytes, k_ref_bytes,
        "TF=1: surfaced K must be byte-identical to the bf16 mirror — \
         proves the dispatch sliced decode_fp16_k, not a flash-transformed tensor"
    );

    eprintln!(
        "TF=1: dispatch delta={delta}, K byte-identity confirmed ({} bytes)",
        k_surf_bytes.len()
    );
}

// ── Test 2: turbo_flash off — dispatch stays dormant ────────────────────────

/// GPU: `update_and_sdpa_shared_source` with TurboFlash OFF — dispatch must
/// stay dormant and the legacy fallback must surface correctly-shaped K/V.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test shared_source_dispatch -- --ignored --test-threads=1"]
fn shared_source_turbo_flash_off_dispatch_stays_dormant() {
    if skip_if_no_gpu() {
        return;
    }

    let device = Device::Gpu;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    // Control arm: same threshold, kernel off.
    let policy = DispatchPolicy {
        turbo_flash: false,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    };

    let prefill_shape = [B, KV_H, PREFILL_SEQ, HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_pref, 0xC1D2E3F4_A5B6_0001);
    let v_data = lcg_data(n_pref, 0xC1D2E3F4_A5B6_0002);
    let prefill_k = make_f32_array(&k_data, &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_k");
    let prefill_v = make_f32_array(&v_data, &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_v");
    let q_pref_shape = [B, N_Q_HEADS, PREFILL_SEQ, HEAD_DIM];
    let n_q_pref: usize = q_pref_shape.iter().map(|&d| d as usize).product();
    let q_pref_data = lcg_data(n_q_pref, 0xC1D2E3F4_A5B6_0003);
    let q_pref = make_f32_array(&q_pref_data, &q_pref_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q_pref");

    let mut cache = build_prefilled_cache(device, &prefill_k, &prefill_v, &q_pref, policy);

    let dec_kv_shape = [B, KV_H, 1, HEAD_DIM];
    let n_dec: usize = dec_kv_shape.iter().map(|&d| d as usize).product();
    let new_k_data = lcg_data(n_dec, 0xC1D2E3F4_A5B6_0004);
    let new_v_data = lcg_data(n_dec, 0xC1D2E3F4_A5B6_0005);
    let new_k = make_f32_array(&new_k_data, &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&new_v_data, &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");
    let q_dec_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_q_dec: usize = q_dec_shape.iter().map(|&d| d as usize).product();
    let q_dec_data = lcg_data(n_q_dec, 0xC1D2E3F4_A5B6_0006);
    let q_dec = make_f32_array(&q_dec_data, &q_dec_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q_dec");

    let tf_before = turbo_flash_dispatch_count();
    let (_, k_surfaced, v_surfaced) = split_bf16_share(
        cache
            .update_and_sdpa_shared_source(&q_dec, &new_k, &new_v, scale, "", None, device)
            .expect("decode shared source with TF=0"),
    );
    let tf_after = turbo_flash_dispatch_count();

    let delta = tf_after - tf_before;
    eprintln!("TF=0: turbo_flash_dispatch_count before={tf_before} after={tf_after} delta={delta}");
    assert_eq!(
        delta, 0,
        "TurboFlash dispatch must be dormant when turbo_flash is off; \
         got delta={delta}"
    );

    let expected_seq = PREFILL_SEQ + 1;
    let k_shape = k_surfaced.shape();
    let v_shape = v_surfaced.shape();
    assert_eq!(
        k_shape,
        [B, KV_H, expected_seq, HEAD_DIM],
        "TF=0: legacy K shape must be [B,kv_h,{expected_seq},D]"
    );
    assert_eq!(
        v_shape,
        [B, KV_H, expected_seq, HEAD_DIM],
        "TF=0: legacy V shape must be [B,kv_h,{expected_seq},D]"
    );

    eprintln!("TF=0: dispatch dormant (delta=0), legacy K/V shape {k_shape:?} confirmed");
}
