//! MSL compile-cache kernel helpers for Gemma4: GeGLU, PLI-GeGLU, softcap, QkNorm variants.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::compile::{compile_shapeless, Closure};
use rmlx_mlx::{
    divide, gelu_tanh, multiply, rms_norm, rope_dynamic, rope_with_freqs_dynamic, scalar_f32, tanh,
    Array, Device, Dtype,
};
use rustc_hash::FxHashMap;
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// geglu_fused — compile_shapeless fusion of `gelu_tanh(gate) * up`
// ---------------------------------------------------------------------------
//
// Reference: mlx-lm/mlx_lm/models/gemma4_text.py:94-96
// @partial(mx.compile, shapeless=True)
// def geglu(gate, x): return nn.gelu_approx(gate) * x
//
// gelu_tanh in rMLX is 8 elementary ops; together with the `* up` multiply
// that's 9 kernel launches per layer's MLP, x 42 layers x every decode step
// on Gemma4 dense. mx.compile fuses these 9 ops into a single Metal program,
// dropping launch count and CPU-side IR rebuild time per step.
//
// Cache: keyed by (in_dtype_tag, device_tag). Closures are shape-agnostic
// (compile_shapeless) so a single compiled Closure handles every
// (batch, seq, hidden) shape we see in decode (1x1xH) AND prefill (1xTxH).

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct GegluKey {
    in_dtype_tag: u8,
    device_tag: u8,
}

fn geglu_dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::F16 => 1,
        Dtype::F32 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I32 => 5,
    }
}

fn geglu_device_tag(d: Device) -> u8 {
    match d {
        Device::Gpu => 0,
        Device::Cpu => 1,
    }
}

static GEGLU_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<GegluKey, std::sync::Arc<Closure>>>> =
    OnceLock::new();

fn geglu_compile_cache() -> &'static Mutex<FxHashMap<GegluKey, std::sync::Arc<Closure>>> {
    GEGLU_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn geglu_get_or_compile(key: GegluKey, device: Device) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = geglu_compile_cache().lock().expect("geglu cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    // Build outside the lock (compile is the slow path).
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 2 {
            return Err(Error::Mlx(format!(
                "geglu_fused closure: expected 2 inputs (gate, up), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let gate = iter.next().expect("gate");
        let up = iter.next().expect("up");
        // `gelu_tanh` returns the gate's dtype (its f32 constants stay internal
        // to it), so this GeGLU is bf16 in, bf16 through the multiply, bf16
        // out. It used to need a cast here because the activation handed back
        // f32 and widened the FFN, the residual stream and the KV cache behind
        // it; that contract now lives in the activation itself.
        let g = gelu_tanh(&gate, device)?;
        let out = multiply(&g, &up, device)?;
        Ok(vec![out])
    });
    let compiled =
        compile_shapeless(raw).map_err(|e| Error::Mlx(format!("geglu compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = geglu_compile_cache().lock().expect("geglu cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute `gelu_tanh(gate) * up` via an mx.compile-fused closure.
///
/// Identical math to the unfused `gelu_tanh(&gate)?` then `multiply(&g, &up)?`
/// pair, but executes as a single compiled Metal program, dropping the 9
/// per-layer pointwise kernel launches to 1.
pub(crate) fn geglu_fused(gate: &Array, up: &Array, device: Device) -> Result<Array> {
    let key = GegluKey {
        in_dtype_tag: geglu_dtype_tag(gate.dtype()),
        device_tag: geglu_device_tag(device),
    };
    let compiled = geglu_get_or_compile(key, device)?;
    let mut outs = compiled.apply(&[gate, up])?;
    outs.pop()
        .ok_or_else(|| Error::Mlx("geglu_fused: closure returned no outputs".to_owned()))
}

// ---------------------------------------------------------------------------
// pli_gelu_fused — compile_shapeless fusion of `gelu_tanh(gate) * per_layer`
// ---------------------------------------------------------------------------
//
// Reference: mlx-lm/mlx_lm/models/gemma4_text.py PerLayerInputGate.__call__
// gate = linear(h); gate = gelu_approx(gate); gate = gate * per_layer
//
// Fuses `gelu_tanh(gate) * per_layer` into one compiled Metal program — same
// pattern as geglu_fused. The two inputs differ only in semantics:
// - geglu_fused: gelu_tanh(gate) * up (both from weight projections)
// - pli_gelu_fused: gelu_tanh(gate) * per_layer (per_layer is the residual tensor)
//
// The gate shape is `[B, T, pli_hidden]` and per_layer is `[B, T, pli_hidden]`
// so the pointwise multiply is element-wise — identical fusion opportunity.
//
// Each PLI forward fires 9 kernel launches (gelu_tanh = 8 ops
// + multiply = 1). Fusing drops this to 1 per layer. On Gemma4 26B-a4b with
// PLI on all 42 decoder layers × every decode step, that is 8 × 42 = 336
// launch savings per step.
//
// Cache: keyed by (in_dtype_tag, device_tag). Shape-agnostic (compile_shapeless)
// so decode (1×1×pli_hidden) and prefill (1×T×pli_hidden) share one closure.

/// Key for the PLI gelu compile cache (same structure as GegluKey).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PliGeluKey {
    in_dtype_tag: u8,
    device_tag: u8,
}

static PLI_GELU_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<PliGeluKey, std::sync::Arc<Closure>>>> =
    OnceLock::new();

fn pli_gelu_compile_cache() -> &'static Mutex<FxHashMap<PliGeluKey, std::sync::Arc<Closure>>> {
    PLI_GELU_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn pli_gelu_get_or_compile(key: PliGeluKey, device: Device) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = pli_gelu_compile_cache()
            .lock()
            .expect("pli_gelu cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    // Build outside the lock (compile is the slow path).
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 2 {
            return Err(Error::Mlx(format!(
                "pli_gelu_fused closure: expected 2 inputs (gate, per_layer), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let gate = iter.next().expect("gate");
        let per_layer = iter.next().expect("per_layer");
        // See geglu_fused: `gelu_tanh` hands back the gate's dtype, so the
        // per-layer-input gating stays at the model dtype without a cast here.
        let g = gelu_tanh(&gate, device)?;
        let out = multiply(&g, &per_layer, device)?;
        Ok(vec![out])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("pli_gelu compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = pli_gelu_compile_cache()
        .lock()
        .expect("pli_gelu cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute `gelu_tanh(gate) * per_layer` via an mx.compile-fused closure.
///
/// Identical math to the unfused `gelu_tanh(&gate)?` then `multiply(&g, &per_layer)?`
/// pair, but executes as a single compiled Metal program. See module-level comment.
pub(crate) fn pli_gelu_fused(gate: &Array, per_layer: &Array, device: Device) -> Result<Array> {
    let key = PliGeluKey {
        in_dtype_tag: geglu_dtype_tag(gate.dtype()),
        device_tag: geglu_device_tag(device),
    };
    let compiled = pli_gelu_get_or_compile(key, device)?;
    let mut outs = compiled.apply(&[gate, per_layer])?;
    outs.pop()
        .ok_or_else(|| Error::Mlx("pli_gelu_fused: closure returned no outputs".to_owned()))
}

// ---------------------------------------------------------------------------
// softcap_fused — compile_shapeless fusion of `tanh(x / cap) * cap`
// ---------------------------------------------------------------------------
//
// Reference: mlx-lm/mlx_lm/models/gemma4_text.py:84-86
// @partial(mx.compile, shapeless=True)
// def logit_softcap(softcap, x): return mx.tanh(x / softcap) * softcap
//
// Final logit softcap fires once per decode step on `[1, 1, vocab]`. For
// gemma-4-26b-a4b vocab=262 144 → 1.05 MB at bf16. Unfused: 3 kernel
// launches (divide, tanh, multiply) × 3 HBM read/write passes ≈ 6.3 MB
// traffic. Fused: 1 launch × 1 read/write ≈ 2.1 MB. Measurement:
// ~0.33 ms/step saved → ~2-3% TPS on Gemma4 dense decode.
//
// Cache: keyed by (in_dtype_tag, device_tag, cap_bits). cap_bits keys on
// the f32 bit pattern of `final_logit_softcapping` so different models with
// different caps each get their own compiled program.
//
// Inputs: 2 (cap, x). Match mlx-lm's signature exactly: cap is passed as a
// scalar Array each call (built from `scalar_f32(cap)`). compile_shapeless
// captures cap via the input list — not the closure environment — so the
// same compiled program is reused across calls with the same dtype/device.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct SoftcapKey {
    in_dtype_tag: u8,
    device_tag: u8,
}

fn softcap_dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::F16 => 1,
        Dtype::F32 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I32 => 5,
    }
}

fn softcap_device_tag(d: Device) -> u8 {
    match d {
        Device::Gpu => 0,
        Device::Cpu => 1,
    }
}

static SOFTCAP_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<SoftcapKey, std::sync::Arc<Closure>>>> =
    OnceLock::new();

fn softcap_compile_cache() -> &'static Mutex<FxHashMap<SoftcapKey, std::sync::Arc<Closure>>> {
    SOFTCAP_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn softcap_get_or_compile(key: SoftcapKey, device: Device) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = softcap_compile_cache()
            .lock()
            .expect("softcap cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 2 {
            return Err(Error::Mlx(format!(
                "softcap_fused closure: expected 2 inputs (cap, x), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let cap = iter.next().expect("cap");
        let x = iter.next().expect("x");
        // tanh(x / cap) * cap — byte-for-byte port of mlx-lm logit_softcap.
        let scaled = divide(&x, &cap, device)?;
        let t = tanh(&scaled, device)?;
        let out = multiply(&t, &cap, device)?;
        Ok(vec![out])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("softcap compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = softcap_compile_cache()
        .lock()
        .expect("softcap cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute `tanh(x / cap) * cap` via an mx.compile-fused closure.
///
/// Byte-for-byte port of mlx-lm `logit_softcap`. `cap` is a Python float
/// in mlx-lm — promoted into the graph as a scalar tensor by tracing.
/// Here we pass it as a 0-D F32 scalar Array.
pub(crate) fn softcap_fused(x: &Array, cap: f32, device: Device) -> Result<Array> {
    let key = SoftcapKey {
        in_dtype_tag: softcap_dtype_tag(x.dtype()),
        device_tag: softcap_device_tag(device),
    };
    let compiled = softcap_get_or_compile(key, device)?;
    // f32-ok: cap is a Python float (weak type) in mlx-lm's logit_softcap — it runs at the
    // activation dtype there. Here cap_arr is a strong F32 scalar, which diverges from the
    // reference in dtype. This is safe because softcap_fused is applied to TERMINAL pre-sampling
    // logits only: the output is never written to the KV cache or the residual stream, so the
    // F32 promotion does not propagate. Changing to astype(x.dtype()) would be numerically
    // equivalent for BF16 logits but is intentionally left as F32 to match the compile key.
    let cap_arr = scalar_f32(cap);
    let mut outs = compiled.apply(&[&cap_arr, x])?;
    outs.pop()
        .ok_or_else(|| Error::Mlx("softcap_fused: closure returned no outputs".to_owned()))
}

// ---------------------------------------------------------------------------
// qk_norm_fused — compile_shapeless fusion of (q rms_norm, k rms_norm)
// ---------------------------------------------------------------------------
//
// QK-norm fusion ported from Qwen3 to Gemma4.
//
// Gemma4 uses plain-gamma RmsNorm (`crate::layers::RmsNorm`), same convention
// as Qwen3. The closure body is identical: `rms_norm(q) + rms_norm(k)`.
// V has its own `RMSNormNoScale` (weight=None) — kept unfused outside this
// helper, runs in parallel anyway.
//
// Skip-on-shared_kv: layers with `shared_kv = Some(...)` reuse upstream K/V
// directly (already normed); only Q runs through `q_norm`. The caller falls
// back to `q_norm.forward(&q, device)` in that branch — qk_norm_fused only
// fires when both Q and K are computed locally.
//
// Cache: keyed by (in_dtype_tag, device_tag, eps_bits). All Gemma4 layers
// share `rms_norm_eps`. compile_shapeless handles every (B, S, H, D) shape.
// Pattern lifted verbatim from `qwen3.rs::qk_norm_fused`.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QkNormKey {
    in_dtype_tag: u8,
    device_tag: u8,
    eps_bits: u32,
}

fn qk_norm_dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bf16 => 0,
        Dtype::F16 => 1,
        Dtype::F32 => 2,
        Dtype::U8 => 3,
        Dtype::U32 => 4,
        Dtype::I32 => 5,
    }
}

fn qk_norm_device_tag(d: Device) -> u8 {
    match d {
        Device::Gpu => 0,
        Device::Cpu => 1,
    }
}

static QK_NORM_COMPILE_CACHE: OnceLock<Mutex<FxHashMap<QkNormKey, std::sync::Arc<Closure>>>> =
    OnceLock::new();

fn qk_norm_compile_cache() -> &'static Mutex<FxHashMap<QkNormKey, std::sync::Arc<Closure>>> {
    QK_NORM_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_get_or_compile(
    key: QkNormKey,
    eps: f32,
    device: Device,
) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = qk_norm_compile_cache()
            .lock()
            .expect("qk_norm cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 4 {
            return Err(Error::Mlx(format!(
                "qk_norm_fused closure: expected 4 inputs (q, k, q_w, k_w), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let q_w = iter.next().expect("q_w");
        let k_w = iter.next().expect("k_w");
        let qn = rms_norm(&q, Some(&q_w), eps, device)?;
        let kn = rms_norm(&k, Some(&k_w), eps, device)?;
        Ok(vec![qn, kn])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("qk_norm compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = qk_norm_compile_cache()
        .lock()
        .expect("qk_norm cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute (rms_norm(q, q_w, eps), rms_norm(k, k_w, eps)) via one compiled
/// closure — fuses the two RMSNorm dispatches per layer per step into a single
/// compiled Metal program. Math identical to two separate `rms_norm` calls.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(crate) fn qk_norm_fused(
    q: &Array,
    k: &Array,
    q_w: &Array,
    k_w: &Array,
    eps: f32,
    device: Device,
) -> Result<(Array, Array)> {
    let key = QkNormKey {
        in_dtype_tag: qk_norm_dtype_tag(q.dtype()),
        device_tag: qk_norm_device_tag(device),
        eps_bits: eps.to_bits(),
    };
    let compiled = qk_norm_get_or_compile(key, eps, device)?;
    let mut outs = compiled.apply(&[q, k, q_w, k_w])?;
    if outs.len() != 2 {
        return Err(Error::Mlx(format!(
            "qk_norm_fused: expected 2 outputs, got {}",
            outs.len()
        )));
    }
    let kn = outs.pop().expect("kn");
    let qn = outs.pop().expect("qn");
    Ok((qn, kn))
}

// ---------------------------------------------------------------------------
// qk_norm_rope_*_fused — extends qk_norm_fused to include the
// post-norm transpose and the RoPE rotation. One compiled Metal program per
// layer-type/dtype, replacing 6 separate dispatches per layer per decode step
// (2 rms_norm + 2 transpose + 2 rope). Two variants because Gemma4's two
// layer types use different RoPE flavours:
// * SlidingAttention -> rope_dynamic (base=rope_theta, freqs computed on-fly)
// * FullAttention -> rope_with_freqs_dynamic (proportional freqs table)
//
// Both rely on `mlx_fast_rope_dynamic` (offset as 0-D i32 Array) so the
// compiled closure is reused across all decode steps — only the offset value
// changes per step. Numerics verified: rope_dynamic matches static rope.
//
// Skip-on-shared_kv: layers with `shared_kv = Some(...)` reuse upstream K/V;
// only Q is normed/roped. The caller continues to use the unfused path on
// that branch.
//
// Pattern lifted from `qwen3.rs::qk_norm_rope_fused`.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QkNormRopeSwaKey {
    in_dtype_tag: u8,
    device_tag: u8,
    eps_bits: u32,
    head_dim: i32,
    rope_theta_bits: u32,
}

static QK_NORM_ROPE_SWA_CACHE: OnceLock<
    Mutex<FxHashMap<QkNormRopeSwaKey, std::sync::Arc<Closure>>>,
> = OnceLock::new();

fn qk_norm_rope_swa_cache() -> &'static Mutex<FxHashMap<QkNormRopeSwaKey, std::sync::Arc<Closure>>>
{
    QK_NORM_ROPE_SWA_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_rope_swa_get_or_compile(
    key: QkNormRopeSwaKey,
    eps: f32,
    head_dim: i32,
    rope_theta: f32,
    device: Device,
) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = qk_norm_rope_swa_cache()
            .lock()
            .expect("qk_norm_rope_swa cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 5 {
            return Err(Error::Mlx(format!(
                "qk_norm_rope_swa_fused closure: expected 5 inputs (q, k, q_w, k_w, offset), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let q_w = iter.next().expect("q_w");
        let k_w = iter.next().expect("k_w");
        let off = iter.next().expect("offset");
        let qn = rms_norm(&q, Some(&q_w), eps, device)?;
        let qt = qn.transpose(&[0, 2, 1, 3], device)?;
        let qr = rope_dynamic(&qt, head_dim, false, rope_theta, 1.0, &off, device)?;
        let kn = rms_norm(&k, Some(&k_w), eps, device)?;
        let kt = kn.transpose(&[0, 2, 1, 3], device)?;
        let kr = rope_dynamic(&kt, head_dim, false, rope_theta, 1.0, &off, device)?;
        Ok(vec![qr, kr])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("qk_norm_rope_swa compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = qk_norm_rope_swa_cache()
        .lock()
        .expect("qk_norm_rope_swa cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Fused (rms_norm → transpose → rope) for Gemma4 SlidingAttention layers.
///
/// Inputs: q,k of shape `[B, S, H, D]` (post-projection, pre-norm).
/// Outputs: q,k of shape `[B, H, S, D]` (post-norm, post-rope).
/// `offset` plumbed through the compiled graph as a 0-D i32 Array.
///
/// **Negative result**: on Gemma4-31b mxfp8 planar (slowest in-scope
/// Gemma4 cell at 11.76 TPS, 62 layers) this fused closure measured +0.03 TPS
/// vs the unfused `qk_norm_fused + transpose + rope` baseline (11.79 vs 11.76,
/// warm). The +0.5 TPS abandon threshold tripped on first cell — call
/// site reverted. Retained as scaffolding (mirrors `qwen3.rs::qk_norm_rope_fused`,
/// also a negative result on Bonsai).
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(crate) fn qk_norm_rope_swa_fused(
    q: &Array,
    k: &Array,
    q_w: &Array,
    k_w: &Array,
    eps: f32,
    head_dim: i32,
    rope_theta: f32,
    offset: i32,
    device: Device,
) -> Result<(Array, Array)> {
    let key = QkNormRopeSwaKey {
        in_dtype_tag: qk_norm_dtype_tag(q.dtype()),
        device_tag: qk_norm_device_tag(device),
        eps_bits: eps.to_bits(),
        head_dim,
        rope_theta_bits: rope_theta.to_bits(),
    };
    let compiled = qk_norm_rope_swa_get_or_compile(key, eps, head_dim, rope_theta, device)?;
    let off_bytes = offset.to_le_bytes();
    let off_arr = Array::from_bytes(&off_bytes, &[], Dtype::I32)?;
    let mut outs = compiled.apply(&[q, k, q_w, k_w, &off_arr])?;
    if outs.len() != 2 {
        return Err(Error::Mlx(format!(
            "qk_norm_rope_swa_fused: expected 2 outputs, got {}",
            outs.len()
        )));
    }
    let kr = outs.pop().expect("kr");
    let qr = outs.pop().expect("qr");
    Ok((qr, kr))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct QkNormRopeFullKey {
    in_dtype_tag: u8,
    device_tag: u8,
    eps_bits: u32,
    head_dim: i32,
}

static QK_NORM_ROPE_FULL_CACHE: OnceLock<
    Mutex<FxHashMap<QkNormRopeFullKey, std::sync::Arc<Closure>>>,
> = OnceLock::new();

fn qk_norm_rope_full_cache() -> &'static Mutex<FxHashMap<QkNormRopeFullKey, std::sync::Arc<Closure>>>
{
    QK_NORM_ROPE_FULL_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn qk_norm_rope_full_get_or_compile(
    key: QkNormRopeFullKey,
    eps: f32,
    head_dim: i32,
    device: Device,
) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = qk_norm_rope_full_cache()
            .lock()
            .expect("qk_norm_rope_full cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 6 {
            return Err(Error::Mlx(format!(
                "qk_norm_rope_full_fused closure: expected 6 inputs (q, k, q_w, k_w, freqs, offset), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let q = iter.next().expect("q");
        let k = iter.next().expect("k");
        let q_w = iter.next().expect("q_w");
        let k_w = iter.next().expect("k_w");
        let freqs = iter.next().expect("freqs");
        let off = iter.next().expect("offset");
        let qn = rms_norm(&q, Some(&q_w), eps, device)?;
        let qt = qn.transpose(&[0, 2, 1, 3], device)?;
        let qr = rope_with_freqs_dynamic(&qt, head_dim, false, 1.0, &off, &freqs, device)?;
        let kn = rms_norm(&k, Some(&k_w), eps, device)?;
        let kt = kn.transpose(&[0, 2, 1, 3], device)?;
        let kr = rope_with_freqs_dynamic(&kt, head_dim, false, 1.0, &off, &freqs, device)?;
        Ok(vec![qr, kr])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("qk_norm_rope_full compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = qk_norm_rope_full_cache()
        .lock()
        .expect("qk_norm_rope_full cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Fused (rms_norm → transpose → rope_with_freqs) for Gemma4 FullAttention.
///
/// Inputs: q,k of shape `[B, S, H, D]` (post-projection, pre-norm).
/// Outputs: q,k of shape `[B, H, S, D]` (post-norm, post-rope).
/// `freqs` is the ProportionalRoPE table built once per layer in the loader.
/// `offset` plumbed through the compiled graph as a 0-D i32 Array.
///
/// Same negative result as `qk_norm_rope_swa_fused`: marked dead-code
/// scaffolding after the call site was reverted. See that helper's docstring.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(crate) fn qk_norm_rope_full_fused(
    q: &Array,
    k: &Array,
    q_w: &Array,
    k_w: &Array,
    eps: f32,
    head_dim: i32,
    freqs: &Array,
    offset: i32,
    device: Device,
) -> Result<(Array, Array)> {
    let key = QkNormRopeFullKey {
        in_dtype_tag: qk_norm_dtype_tag(q.dtype()),
        device_tag: qk_norm_device_tag(device),
        eps_bits: eps.to_bits(),
        head_dim,
    };
    let compiled = qk_norm_rope_full_get_or_compile(key, eps, head_dim, device)?;
    let off_bytes = offset.to_le_bytes();
    let off_arr = Array::from_bytes(&off_bytes, &[], Dtype::I32)?;
    let mut outs = compiled.apply(&[q, k, q_w, k_w, freqs, &off_arr])?;
    if outs.len() != 2 {
        return Err(Error::Mlx(format!(
            "qk_norm_rope_full_fused: expected 2 outputs, got {}",
            outs.len()
        )));
    }
    let kr = outs.pop().expect("kr");
    let qr = outs.pop().expect("qr");
    Ok((qr, kr))
}

// ---------------------------------------------------------------------------
// Per-layer input gating
// ---------------------------------------------------------------------------
