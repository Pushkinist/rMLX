//! Deterministic load-time MSL precompile for KV codecs.
//!
//! Custom Metal kernels in this crate compile **lazily** — `MetalKernel::new`
//! only registers the kernel; MLX compiles the MSL → Metal pipeline on the
//! *first* `apply()` dispatch (see `docs/FFI.md` § `MetalKernel`). For KV codecs
//! that dispatch a real Metal kernel on the hot path, that first dispatch lands
//! inside the first user request — so the first forward pays a one-time shader
//! cold-compile that a 1-token `"hi"` warmup never triggers (the codec kernels
//! only fire on a real prefill encode). On large models this reads as a
//! multi-second stall, and an orchestrator with a request-level timeout can
//! misreport it as a load failure.
//!
//! This module warms those kernels at model-load / codec-attach time with one
//! representative dispatch at a real shape, so the compile cost is paid during
//! the deterministic preload window rather than on the first request.
//!
//! The warm is **general per-codec**, keyed off [`KvQuant::carries_msl`] — never
//! an arch name. It is a no-op for `none` and for the CPU-hot-path codecs (the
//! V-only iso / rotor families and QJL-on rotor-K, see
//! [`KvQuant::cpu_hot_path_reason`]) whose production encode + dequant run on
//! CPU and therefore have no q8 shader to warm here. The K-only iso / rotor
//! codecs ARE Metal on the hot path but dispatch the iso/rotor MSL kernel for K
//! (not the shared q8_0 K kernel this module warms), so they are also skipped
//! ([`KvQuant::is_k_only_iso_rotor`]) and compile lazily on first prefill.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use crate::KvQuant;

/// Warm every Metal kernel a KV codec dispatches on its hot path, so the
/// shader cold-compile is paid here (load time) instead of inside the first
/// user request.
///
/// `head_dim` and `kv_heads` come from the model config so the warm dispatch
/// uses the production shape (kernel template specialization is shape-aware).
///
/// Behaviour:
/// - `device != Gpu` → no-op (CPU runs have no Metal pipeline to compile).
/// - codec does not carry MSL (`none`) → no-op.
/// - CPU-hot-path codec (V-only iso / rotor, QJL-on rotor-K) → no-op (its
///   encode/dequant is CPU; the GPU fused-QK encoder, when present, is opt-in
///   via `--fused-qk` and not on the default path).
/// - K-only iso / rotor codec (`is_k_only_iso_rotor`) → no-op here (Metal on the
///   hot path, but its K kernel is the iso/rotor MSL kernel, not the shared q8_0
///   K kernel this module warms; it compiles lazily on first prefill).
/// - otherwise → warm the shared q8_0 K-side kernels (every MSL-carrying KV
///   codec quantizes K with q8_0) plus, where applicable, the codec's V-side
///   GPU kernel.
///
/// Best-effort: a warm dispatch failure is logged at `warn!` and returns
/// `Ok(())` — a failed precompile must not abort model load (the kernel will
/// simply compile lazily on first use, the previous lazy-compile behaviour).
#[allow(
    clippy::cognitive_complexity,
    reason = "linear sequence of early-return guards (device / carries_msl / cpu_hot_path / shape-alignment) plus two best-effort warm calls; splitting the guards into helpers would add indirection without reducing local complexity"
)]
pub fn precompile_kv_codec_msl(
    kq: KvQuant,
    head_dim: usize,
    kv_heads: usize,
    device: Device,
) -> Result<()> {
    if device != Device::Gpu {
        return Ok(());
    }
    if head_dim == 0 {
        // Degenerate config (arch did not report head_dim). The shape math
        // below would build a 0-element warm array and dispatch q8 on an empty
        // buffer — a silent, useless dispatch. Skip; the kernel compiles lazily
        // on first real use.
        tracing::debug!(kv_quant = %kq, "precompile_kv_codec_msl: skip — head_dim unknown (0)");
        return Ok(());
    }
    if !kq.carries_msl() {
        tracing::debug!(kv_quant = %kq, "precompile_kv_codec_msl: codec carries no MSL — skip");
        return Ok(());
    }
    if let Some(reason) = kq.cpu_hot_path_reason() {
        tracing::debug!(
            kv_quant = %kq,
            reason,
            "precompile_kv_codec_msl: CPU-hot-path codec — no shader to warm, skip"
        );
        return Ok(());
    }
    // K-only iso / rotor codecs are Metal on the hot path (so
    // `cpu_hot_path_reason()` is `None`), but their K side is the iso/rotor MSL
    // kernel, NOT the shared q8_0 K kernel that `warm_q8` below compiles. Warming
    // q8 for them would compile the wrong shader and miss the real one; their
    // iso/rotor K kernel compiles lazily on first prefill (lazy-compile path).
    if kq.is_k_only_iso_rotor() {
        tracing::debug!(
            kv_quant = %kq,
            "precompile_kv_codec_msl: K-only iso/rotor codec — K kernel is iso/rotor MSL, \
             not q8; lazy-compile on first prefill, skip q8 warm"
        );
        return Ok(());
    }

    let kv_heads_i = kv_heads.max(1) as i32;
    let head_dim_i = head_dim.max(1) as i32;
    // q8_0 (the shared K-side kernel) requires total elements to be a multiple
    // of its group size (128). TurboQuant/PlanarQuant V kernels group by 32 and
    // additionally need `head_dim % 32 == 0`. Pick a token count that makes the
    // per-(kv_head, head_dim) buffer a multiple of 128 so every warm dispatch is
    // shape-legal; keep it tiny so load-time cost is the compile, not the data.
    let per_token = kv_heads * head_dim.max(1);
    let warm_tokens: i32 = if per_token == 0 {
        128
    } else {
        // smallest n_tokens with (per_token * n_tokens) % 128 == 0, capped small.
        let lcm = lcm_usize(per_token, 128);
        (lcm / per_token).clamp(1, 32) as i32
    };
    let shape = [1, kv_heads_i, warm_tokens, head_dim_i];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    if !n.is_multiple_of(128) {
        // head_dim/kv_heads geometry can't be padded to a legal q8 group here;
        // skip rather than dispatch an out-of-spec kernel.
        tracing::debug!(
            kv_quant = %kq, head_dim, kv_heads,
            "precompile_kv_codec_msl: warm shape not group-aligned — skip (lazy compile on first use)"
        );
        return Ok(());
    }

    let t0 = std::time::Instant::now();
    let warm = match Array::from_bytes(&vec![0u8; n * 2], &shape, Dtype::Bf16) {
        Ok(a) => a,
        Err(e) => {
            tracing::warn!(error = %e, kv_quant = %kq, "precompile_kv_codec_msl: warm array alloc failed — skipping");
            return Ok(());
        }
    };

    // K-side q8_0: shared by every MSL-carrying KV codec (K8V*/Planar*/Turbo*/
    // RotKTq4V all quantize K with q8_0, group=128). Warm both directions.
    if let Err(e) = warm_q8(&warm, device) {
        tracing::warn!(error = %e, kv_quant = %kq, "precompile_kv_codec_msl: q8 K-side warm failed (non-fatal)");
    }

    // V-side codec kernel, where the codec dispatches one on the hot path.
    if let Err(e) = warm_v_side(kq, &warm, device) {
        tracing::warn!(error = %e, kv_quant = %kq, "precompile_kv_codec_msl: V-side warm failed (non-fatal)");
    }

    tracing::info!(
        kv_quant = %kq,
        head_dim,
        kv_heads,
        warm_ms = t0.elapsed().as_millis(),
        "precompile_kv_codec_msl: MSL warm complete"
    );
    Ok(())
}

/// Warm the q8_0 quantize + dequantize Metal kernels by round-tripping a small
/// array. Forces both pipelines to compile.
fn warm_q8(warm: &Array, device: Device) -> Result<()> {
    let (codes, scales) = crate::q8_msl::q8_quantize_gpu(warm, device)?;
    codes.eval()?;
    scales.eval()?;
    let total: i32 = warm.shape().iter().product();
    let recovered =
        crate::q8_msl::q8_dequantize_gpu(&codes, &scales, &[total], Dtype::Bf16, device)?;
    recovered.eval()?;
    Ok(())
}

/// Least common multiple of two positive `usize`s (saturating; returns the
/// product clamped to `usize::MAX` on overflow, which is harmless here because
/// callers only use the ratio `lcm / a`).
fn lcm_usize(a: usize, b: usize) -> usize {
    if a == 0 || b == 0 {
        return 0;
    }
    let g = gcd_usize(a, b);
    (a / g).saturating_mul(b)
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Warm the V-side GPU kernel for codecs that dispatch one on the production
/// hot path. Only the TurboQuant-4 and PlanarQuant-4 V kernels qualify today;
/// other V codecs either share the q8 path (K8V8) or run their V encode on CPU.
///
/// The TurboQuant/PlanarQuant V kernels require `head_dim % 32 == 0`; when the
/// warm buffer's last axis is not 32-aligned the V warm is skipped (the kernel
/// will compile lazily on first real use, same as the lazy-compile path).
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "the wildcard arm is the contract: codecs whose V encode is q8 (K8V8), CPU-forced (turbo3/turbo2/tcq), or CPU-only (iso/rotor, skipped before this call) have no extra V shader to warm here. Exhaustive expansion would add a no-op arm per future variant for no semantic gain."
)]
fn warm_v_side(kq: KvQuant, warm: &Array, device: Device) -> Result<()> {
    let head_dim = warm.shape().last().copied().unwrap_or(0);
    if head_dim <= 0 || head_dim % 32 != 0 {
        return Ok(());
    }
    match kq {
        // tq4 V — K8V4 and RotKTq4V both encode V with TurboQuant 4-bit on GPU.
        KvQuant::K8V4 | KvQuant::RotKTq4V => {
            let (codes, scales) = crate::turboquant_msl::turbo_quantize_v4_gpu(warm, device)?;
            codes.eval()?;
            scales.eval()?;
        }
        // PlanarQuant-4 V — Planar codec encodes V with the Givens-rotation
        // 4-bit kernel on GPU.
        KvQuant::Planar => {
            let (codes, scales, rotations) =
                crate::planarquant_msl::planar_quantize_v4_gpu(warm, device)?;
            codes.eval()?;
            scales.eval()?;
            rotations.eval()?;
        }
        // PlanarQuant-3 V — same Givens-rotation kernel family at 3-bit.
        // (quant_planar_v.rs dispatches planar_quantize_v3_gpu on the GPU
        // append path when bits == 3.)
        KvQuant::Planar3 => {
            let (codes, scales, rotations) =
                crate::planarquant_msl::planar_quantize_v3_gpu(warm, device)?;
            codes.eval()?;
            scales.eval()?;
            rotations.eval()?;
        }
        // K8V8 V side reuses the q8 kernel already warmed above; all other
        // MSL-carrying codecs run their V encode on CPU (turbo3/turbo2/tcq)
        // so there is no extra V shader to warm here.
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
#[path = "precompile_tests.rs"]
mod precompile_tests;
