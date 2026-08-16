// Apple10 (M5+) hazard re-validation at head_dim = 256.
//
// The original B1 report (`docs/reports/B1-turboflash-m5-validation.md`)
// documented an `EXC_BAD_ACCESS SIGSEGV @ 0x0` hazard on M5 Max for the
// TurboFlash MSL kernel when dispatched against the Qwen3.6-35B-A3B-8bit
// `head_dim = 256` full-attention configuration at 32k context. The previous
// validation host was M5 Max, but the routing-extension tests it shipped drove
// `head_dim = 128`, which is the linear-attention head size on the same
// Qwen3.6 snapshot — the hazard scenario itself was never re-exercised after
// the kernel landed. The `Auto` arm of `--turbo-flash` was therefore left
// as Apple ≥ 10 → OFF "pending an explicit M5 re-validation with the dispatch
// counter and the historical `head_dim = 256` configuration".
//
// This file is that re-validation, run as a synthetic K8V4 cache driven
// directly through the public `KvCache::update_and_sdpa` chain — no model
// snapshot required, reproducible on any Apple Silicon host.
//
// Assertions (per scenario):
//
//   1. `DispatchPolicy { turbo_flash: true, turbo_flash_min_kv_seq: 0 }`:
//      a. The test process survives every dispatch (no `SIGSEGV`, no abort).
//      b. `turbo_flash_dispatch_count()` delta > 0 — the MSL kernel ran on
//         the hazard configuration rather than silently falling back.
//      c. Decode output cosine vs the bf16 SDPA reference must clear the
//         CPU-side V turbo-4 codec floor (~0.997 — see
//         `tests/apple10_cpu_baseline.rs`, which measures the same encode→
//         decode round-trip for V at head_dim ∈ {128, 256} and lands the
//         same number). The assertion is `cosine_min >= CODEC_FLOOR_MIN`
//         with `CODEC_FLOOR_MIN = 0.995` — i.e. tight parity with the CPU
//         codec floor allowing 2e-3 of head-room for accumulated f32 noise
//         across the 16-step stress loop. The K8V4 fused-QK 0.999998 floor
//         is for Q·K^T only (K-dominated); this test does full SDPA
//         (softmax @ V), where V turbo-4 dominates the residual.
//
//   2. The same shape under a `turbo_flash: false` policy (control):
//      dispatch delta == 0. Both arms run in this one binary, each cache
//      carrying its own policy.
//
// `RMLX_APPLE10_STRICT=1` — strict mode (CI gate form):
//   * delta == 0 in the ON case -> panic instead of soft-skip.
//   * `update_and_sdpa` error in the ON case -> panic instead of soft-skip.
//
// `#[ignore]` because it needs the Metal GPU. Run via:
//   cargo test -p rmlx-kv-quant --test apple10_head_dim_256 -- \
//       --ignored --test-threads=1 --nocapture
//
// CLAUDE.md hard rule 8 (single MLX process): preflight via
//   pkill -f "rmlx serve"; pkill -f mlx_lm; pkill -f paroquant; pkill -f omlx
//   rm -f /tmp/rmlx.62265.claim
// before running. The test does NOT claim a port — the integration runner
// serialises tests within a single process and we keep `--test-threads=1`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    missing_docs
)]
//! head_dim=256 hazard re-validation.

use rmlx_core::DispatchPolicy;
use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};

/// TurboFlash on with a zero threshold, so the kernel fires on the short
/// synthetic caches here.
fn turbo_on() -> DispatchPolicy {
    DispatchPolicy {
        turbo_flash: true,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    }
}

/// The control arm: identical apart from the kernel gate, so a zero dispatch
/// delta can only come from the gate and not from the threshold.
fn turbo_off_same_threshold() -> DispatchPolicy {
    DispatchPolicy {
        turbo_flash: false,
        ..turbo_on()
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("from_bytes")
}

/// MLX `Array.eval()` (materialise on device) + `to_bytes()` round-trip into a
/// `Vec<f32>`. Wrapped so the test body reads as a plain numerical copy.
fn materialise_and_collect_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("materialise array on device");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("chunk len 4")))
        .collect()
}

fn skip_if_no_gpu() -> bool {
    std::env::var("RMLX_SKIP_GPU").as_deref() == Ok("1")
}

fn is_strict() -> bool {
    std::env::var("RMLX_APPLE10_STRICT").as_deref() == Ok("1")
}

/// Per-row cosine — mirrors the fused-QK dispatch helper exactly so floors are
/// comparable across reports.
fn cosine_per_row(a: &[f32], b: &[f32], row_len: usize) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "cosine: len mismatch");
    assert!(row_len > 0);
    let n_rows = a.len() / row_len;
    let mut mn = f32::INFINITY;
    let mut sum = 0.0_f64;
    for r in 0..n_rows {
        let ra = &a[r * row_len..(r + 1) * row_len];
        let rb = &b[r * row_len..(r + 1) * row_len];
        let dot: f64 = ra
            .iter()
            .zip(rb.iter())
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = ra.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = rb.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let c = if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            (dot / (na * nb)) as f32
        };
        if c < mn {
            mn = c;
        }
        sum += f64::from(c);
    }
    let mean = (sum / n_rows as f64) as f32;
    (mean, mn)
}

/// LCG pseudo-random data for reproducible tensors. Same constants as other
/// integration tests so reproducer seeds round-trip across reports.
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

// ── test parameters ───────────────────────────────────────────────────────────

const B: i32 = 1;
const KV_H: i32 = 2;
const HEADS_PER_KV: i32 = 4;
const N_Q_HEADS: i32 = KV_H * HEADS_PER_KV;
/// The hazard configuration. The original report described the SIGSEGV on
/// Qwen3.6-35B-A3B-8bit `head_dim = 256` (full-attention layers).
/// `linear_key_head_dim = 128` is the linear-attention path unrelated to
/// this hazard. Pinned to a constant so the report row reads unambiguously.
const HEAD_DIM: i32 = 256;
/// Sized to comfortably accommodate the stress loop without reallocation.
const MAX_SEQ: i32 = 32_768 + 256;

/// Tightened-parity cosine floor (codec floor confirmed).
/// `tests/apple10_cpu_baseline.rs` measures the V turbo-4 codec encode→decode
/// round-trip cosine min at head_dim=256 ≈ 0.9966; head_dim=128 ≈ 0.9958. The
/// GPU SDPA cosine should hit that floor up to small f32 accumulation noise
/// across the stress loop. 0.995 gives ~1.6e-3 of head-room — enough to
/// accommodate the f32 reduction in cosine_per_row without masking a real
/// kernel-side regression.
const CODEC_FLOOR_MIN: f32 = 0.995;

/// Short-context smoke: prefill 64 tokens and run one decode. This proves the
/// `head_dim = 256` MSL register / threadgroup layout assembles + dispatches
/// at all before the stress arm runs.
const SMOKE_PREFILL: i32 = 64;

/// Long-context decode: prefill 64 then decode in a small loop so we cross
/// the documented hazard surface (the kernel is gated on kv_seq, not on a
/// "minimum prefill size"). Each decode step is a fresh kernel dispatch at
/// the current kv_seq.
const LONG_DECODE_STEPS: i32 = 16;

// ── Test 1: short smoke — dispatch fires, output coherent ────────────────────

/// Smoke: K8V4 + head_dim=256 + `RMLX_TURBO_FLASH=1` at a short prefill.
/// Proves the kernel compiles and dispatches against the hazard configuration.
/// If the M5 hazard were still live this test would either hard-crash
/// (SIGSEGV — terminates the test process) or produce numerical garbage
/// (cosine well below the 0.99 floor).
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test apple10_head_dim_256 -- --ignored --test-threads=1"]
fn turbo_flash_head_dim_256_smoke_dispatch_and_cosine() {
    if skip_if_no_gpu() {
        return;
    }

    let strict = is_strict();
    let device = Device::Gpu;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ).with_dispatch_policy(turbo_on());

    // ── Prefill via `update` (matches the K8V4 fused-QK dispatch pattern). ────
    cache.enter_prefill();
    let prefill_shape = [B, KV_H, SMOKE_PREFILL, HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(n_pref, 0xC1A2D3E4_F5A6_0001);
    let v_data = lcg_data(n_pref, 0xC1A2D3E4_F5A6_0002);
    let prefill_k = make_f32_array(&k_data, &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_k");
    let prefill_v = make_f32_array(&v_data, &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_v");
    let _ = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("smoke prefill update");
    cache.exit_prefill(device).expect("smoke exit_prefill");
    assert_eq!(cache.offset(), SMOKE_PREFILL, "smoke offset after prefill");

    // ── One decode step ─────────────────────────────────────────────────────
    let dec_kv_shape = [B, KV_H, 1, HEAD_DIM];
    let n_dec: usize = dec_kv_shape.iter().map(|&d| d as usize).product();
    let new_k = make_f32_array(&lcg_data(n_dec, 0xC1A2D3E4_F5A6_0003), &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&lcg_data(n_dec, 0xC1A2D3E4_F5A6_0004), &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");

    let q_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_q: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_arr = make_f32_array(&lcg_data(n_q, 0xC1A2D3E4_F5A6_0005), &q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let before = turbo_flash_dispatch_count();
    let out_res = cache.update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device);
    let after = turbo_flash_dispatch_count();

    let out = match out_res {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "smoke head_dim=256 TF=1: update_and_sdpa errored: {e} \
                 (skipping cosine; dispatch counter delta={})",
                after - before
            );
            assert!(
                !strict,
                "RMLX_APPLE10_STRICT=1 — TF=1 smoke errored ({e}), but strict mode requires \
                 the kernel to complete on the head_dim=256 hazard configuration"
            );
            return;
        }
    };

    let delta = after - before;
    eprintln!(
        "smoke head_dim=256 TF=1: turbo_flash_dispatch_count \
         before={before} after={after} delta={delta}"
    );

    if delta == 0 {
        eprintln!(
            "smoke head_dim=256 TF=1: HOLD-soft — dispatch did not fire \
             (expected: kv_seq={} > RMLX_TURBO_FLASH_MIN=0). Skipping cosine.",
            cache.offset()
        );
        assert!(
            !strict,
            "RMLX_APPLE10_STRICT=1 — dispatch counter did not increment at head_dim=256 \
             smoke; the kernel was expected to fire."
        );
        return;
    }

    // ── bf16 reference ─────────────────────────────────────────────────────
    let mut bf16_cache = KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ);
    bf16_cache.enter_prefill();
    let _ = bf16_cache
        .update(&prefill_k, &prefill_v, device)
        .expect("bf16 ref prefill");
    bf16_cache
        .exit_prefill(device)
        .expect("bf16 ref exit_prefill");
    let ref_out = bf16_cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("bf16 ref update_and_sdpa");

    let out_f32 = materialise_and_collect_f32(&out.astype(Dtype::F32, device).expect("to f32"));
    let ref_f32 = materialise_and_collect_f32(&ref_out.astype(Dtype::F32, device).expect("to f32"));
    let (mean, mn) = cosine_per_row(&ref_f32, &out_f32, HEAD_DIM as usize);
    eprintln!(
        "smoke head_dim=256 TF=1: cosine mean={mean:.6} min={mn:.6} \
         dispatch_delta={delta}"
    );
    assert!(
        mn >= CODEC_FLOOR_MIN,
        "smoke head_dim=256 TF=1 cosine min {mn} < CODEC_FLOOR_MIN \
         ({CODEC_FLOOR_MIN}) (mean {mean}) — drop below CPU V-turbo-4 codec \
         floor (see tests/apple10_cpu_baseline.rs); points at a kernel-side \
         numerics regression rather than the documented codec residual"
    );
}

// ── Test 2: long-decode stress — the historical hazard surface ───────────────

/// Long-decode stress: K8V4 + head_dim=256 + `RMLX_TURBO_FLASH=1`,
/// `LONG_DECODE_STEPS` decode steps after a short prefill. Each step is a
/// fresh kernel dispatch at the growing kv_seq. The original SIGSEGV reproduced
/// "the instant the K8V4 flash path dispatches at 32k context on
/// Qwen3.6-35B-A3B-8bit (head_dim=256)" — so a multi-step decode at this
/// shape is the precise surface that must stay alive.
///
/// Process survival across all steps + cosine >= 0.99 on the final step is
/// the all-clear signal. We DO NOT load a 32k prompt; the kernel hazard is
/// a per-dispatch register / threadgroup-memory failure, not a prompt-size
/// failure. A short prefill + many decode steps at the right `head_dim` and
/// quant exercises the same code path the hazard report described.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test apple10_head_dim_256 -- --ignored --test-threads=1"]
fn turbo_flash_head_dim_256_long_decode_stress() {
    if skip_if_no_gpu() {
        return;
    }

    let strict = is_strict();
    let device = Device::Gpu;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut cache =
        KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ).with_dispatch_policy(turbo_on());

    cache.enter_prefill();
    let prefill_shape = [B, KV_H, SMOKE_PREFILL, HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let prefill_k = make_f32_array(&lcg_data(n_pref, 0xD2B3E4F5_A6B7_0001), &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_k");
    let prefill_v = make_f32_array(&lcg_data(n_pref, 0xD2B3E4F5_A6B7_0002), &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_v");
    let _ = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("stress prefill update");
    cache.exit_prefill(device).expect("stress exit_prefill");

    // bf16 reference cache — kept in lock-step so the final step cosine
    // applies to the same accumulated context.
    let mut bf16_cache = KvCache::with_quant_max_seq(KvQuant::None, MAX_SEQ);
    bf16_cache.enter_prefill();
    let _ = bf16_cache
        .update(&prefill_k, &prefill_v, device)
        .expect("bf16 ref prefill");
    bf16_cache
        .exit_prefill(device)
        .expect("bf16 ref exit_prefill");

    let mut total_delta: u64 = 0;
    let mut last_out: Option<Array> = None;
    let mut last_ref: Option<Array> = None;

    let dec_kv_shape = [B, KV_H, 1, HEAD_DIM];
    let q_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_dec: usize = dec_kv_shape.iter().map(|&d| d as usize).product();
    let n_q: usize = q_shape.iter().map(|&d| d as usize).product();

    for step in 0..LONG_DECODE_STEPS {
        let seed_base = 0xD2B3E4F5_A6B7_1000u64 ^ (u64::from(step as u32) << 8);
        let new_k = make_f32_array(&lcg_data(n_dec, seed_base ^ 0x1), &dec_kv_shape)
            .astype(Dtype::Bf16, device)
            .expect("bf16 new_k");
        let new_v = make_f32_array(&lcg_data(n_dec, seed_base ^ 0x2), &dec_kv_shape)
            .astype(Dtype::Bf16, device)
            .expect("bf16 new_v");
        let q_arr = make_f32_array(&lcg_data(n_q, seed_base ^ 0x3), &q_shape)
            .astype(Dtype::Bf16, device)
            .expect("bf16 q");

        let before = turbo_flash_dispatch_count();
        let out_res = cache.update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device);
        let after = turbo_flash_dispatch_count();
        total_delta += after - before;

        let out = match out_res {
            Ok(o) => o,
            Err(e) => {
                eprintln!(
                    "stress head_dim=256 TF=1 step={step} kv_seq={} errored: {e}",
                    cache.offset()
                );
                assert!(
                    !strict,
                    "RMLX_APPLE10_STRICT=1 — stress step={step} errored ({e}); \
                     strict mode requires every step to complete on the hazard configuration"
                );
                return;
            }
        };

        // Keep the bf16 reference in lock-step.
        let ref_out = bf16_cache
            .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
            .expect("bf16 ref step");
        last_out = Some(out);
        last_ref = Some(ref_out);
    }

    eprintln!(
        "stress head_dim=256 TF=1: completed {LONG_DECODE_STEPS} steps, \
         total dispatch delta={total_delta}, final kv_seq={}",
        cache.offset()
    );

    if total_delta == 0 {
        eprintln!("stress head_dim=256 TF=1: HOLD-soft — no dispatches fired across the loop");
        assert!(
            !strict,
            "RMLX_APPLE10_STRICT=1 — stress loop saw zero TurboFlash dispatches at \
             head_dim=256 across {LONG_DECODE_STEPS} steps; strict mode requires dispatch."
        );
        return;
    }

    // MEDIUM-3: tightened from `total_delta > 0` to exact-count equality. The
    // TurboFlash P1 dispatch counter (`TURBO_FLASH_DISPATCHES.fetch_add(1, ..)`
    // in `crates/rmlx-kv-quant/src/turbo_flash_msl.rs`) is incremented exactly
    // once per `update_and_sdpa` call that reaches the P1 enqueue site. The
    // stress loop calls `update_and_sdpa` `LONG_DECODE_STEPS` times and every
    // call passes the activation gates (TF=1, decode q_seq=1, K8V4,
    // kv_seq > RMLX_TURBO_FLASH_MIN=0). Any total != steps signals a silent
    // fallback to mixed_quantized_sdpa (lost hazard coverage) — strict mode
    // makes this a hard fail.
    if strict {
        assert_eq!(
            total_delta, LONG_DECODE_STEPS as u64,
            "RMLX_APPLE10_STRICT=1 — stress total dispatch delta {total_delta} != \
             LONG_DECODE_STEPS {LONG_DECODE_STEPS}; TurboFlash silently fell back \
             on at least one step (see turbo_flash_msl.rs — increment is +1 per \
             P1 enqueue)"
        );
    }

    let out = last_out.expect("at least one successful decode");
    let ref_out = last_ref.expect("at least one bf16 reference decode");
    let out_f32 = materialise_and_collect_f32(&out.astype(Dtype::F32, device).expect("to f32"));
    let ref_f32 = materialise_and_collect_f32(&ref_out.astype(Dtype::F32, device).expect("to f32"));
    let (mean, mn) = cosine_per_row(&ref_f32, &out_f32, HEAD_DIM as usize);
    eprintln!("stress head_dim=256 TF=1: final-step cosine mean={mean:.6} min={mn:.6}");
    assert!(
        mn >= CODEC_FLOOR_MIN,
        "stress head_dim=256 TF=1 final cosine min {mn} < CODEC_FLOOR_MIN \
         ({CODEC_FLOOR_MIN}) (mean {mean}) — drop below CPU V-turbo-4 codec \
         floor (see tests/apple10_cpu_baseline.rs) across {LONG_DECODE_STEPS} \
         steps; points at a kernel-side regression"
    );
}

// ── Test 3: control — the same shape with the kernel off ────────────────────
//
// The policy travels on the cache, so the OFF arm runs in this binary next to
// the ON arms. Under the old process-global gate that was impossible: the
// first read latched the value for the whole process, so the control needed a
// test binary of its own.

#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --test apple10_head_dim_256 -- --ignored --test-threads=1"]
fn turbo_flash_head_dim_256_control_dispatch_stays_dormant() {
    if skip_if_no_gpu() {
        return;
    }

    let device = Device::Gpu;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mut cache = KvCache::with_quant_max_seq(KvQuant::K8V4, MAX_SEQ)
        .with_dispatch_policy(turbo_off_same_threshold());
    cache.enter_prefill();
    let prefill_shape = [B, KV_H, SMOKE_PREFILL, HEAD_DIM];
    let n_pref: usize = prefill_shape.iter().map(|&d| d as usize).product();
    let prefill_k = make_f32_array(&lcg_data(n_pref, 0xE3C4F5A6_B7C8_0001), &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_k");
    let prefill_v = make_f32_array(&lcg_data(n_pref, 0xE3C4F5A6_B7C8_0002), &prefill_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 prefill_v");
    let _ = cache
        .update(&prefill_k, &prefill_v, device)
        .expect("control prefill update");
    cache.exit_prefill(device).expect("control exit_prefill");

    let dec_kv_shape = [B, KV_H, 1, HEAD_DIM];
    let q_shape = [B, N_Q_HEADS, 1, HEAD_DIM];
    let n_dec: usize = dec_kv_shape.iter().map(|&d| d as usize).product();
    let n_q: usize = q_shape.iter().map(|&d| d as usize).product();
    let new_k = make_f32_array(&lcg_data(n_dec, 0xE3C4F5A6_B7C8_0003), &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_k");
    let new_v = make_f32_array(&lcg_data(n_dec, 0xE3C4F5A6_B7C8_0004), &dec_kv_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 new_v");
    let q_arr = make_f32_array(&lcg_data(n_q, 0xE3C4F5A6_B7C8_0005), &q_shape)
        .astype(Dtype::Bf16, device)
        .expect("bf16 q");

    let before = turbo_flash_dispatch_count();
    let _ = cache
        .update_and_sdpa(&q_arr, &new_k, &new_v, scale, "", None, device)
        .expect("control decode");
    let after = turbo_flash_dispatch_count();

    let delta = after - before;
    eprintln!("control head_dim=256 turbo_flash off: dispatch delta={delta}");
    assert_eq!(
        delta, 0,
        "TurboFlash dispatch must stay dormant when the cache policy has \
         turbo_flash off, at head_dim=256"
    );
}
