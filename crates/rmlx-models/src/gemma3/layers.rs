//! Gemma3-local layer types.
//!
//! `Linear`, `Embedding`, and `Mlp` are defined locally (not imported from
//! layers.rs) because the medgemma affine snapshot requires `.biases` support
//! in quantized_matmul, which the shared layers::Linear does not carry.
//!
//! `RmsNormShifted` (the Gemma3-specific gamma+1 convention) is **re-exported
//! from `rmlx-runtime`** — the type definition there is byte-identical to the
//! previous local copy.

#![allow(clippy::struct_field_names)]
use rustc_hash::FxHashMap;
use std::sync::{Mutex, OnceLock};

use rmlx_core::error::{Error, Result};
use rmlx_mlx::compile::{compile_shapeless, Closure};
use rmlx_mlx::{add, gelu_tanh, multiply, rms_norm, Array, Device, Dtype};

// `RmsNormShifted` lives in `rmlx-runtime`. We re-export it under the same
// `pub(super)` name so the rest of the gemma3 module compiles unchanged.
pub(super) use rmlx_runtime::RmsNormShifted;

// ---------------------------------------------------------------------------
// Local Linear + Embedding with biases support
// ---------------------------------------------------------------------------
//
// The medgemma affine snapshot stores `.biases` siblings alongside `.scales`.
// `layers::Linear::Quantized` always passes `None` for biases (mxfp8 path
// doesn't need them). Defining local versions here avoids touching layers.rs.
//
// These types mirror `layers::Linear` / `layers::Embedding` but add an
// optional `biases: Option<Array>` field that is forwarded to `quantized_matmul`.

#[allow(missing_debug_implementations)]
pub(super) enum Linear {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Linear {
    #[allow(dead_code)]
    pub(super) fn try_clone(&self) -> Result<Self> {
        match self {
            Linear::Plain { weight } => Ok(Linear::Plain {
                weight: weight.try_clone()?,
            }),
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => Ok(Linear::Quantized {
                weight: weight.try_clone()?,
                scales: scales.try_clone()?,
                biases: biases.as_ref().map(Array::try_clone).transpose()?,
                group_size: *group_size,
                bits: *bits,
                mode: mode.clone(),
            }),
        }
    }

    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Linear::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
        }
    }
}

#[allow(missing_debug_implementations)]
pub(super) enum Embedding {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Embedding {
    /// Look up token ids. `ids` shape: `[seq]` I32.
    pub(super) fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => weight.take(ids, 0, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => gemma3_embedding_lookup(
                ids,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                device,
            ),
        }
    }

    /// Treat embedding as a linear layer for lm_head (tied-weights output projection).
    pub(super) fn as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
        }
    }
}

/// On-device quantized embedding lookup with optional biases.
///
/// Mirrors `mlx_lm.nn.QuantizedEmbedding.__call__`:
/// `dequantize(weight[ids], scales[ids], biases[ids], …)`
///
/// Earlier versions ran this through `Device::Cpu` with an `eye(seq) @ w`
/// trick, forcing a GPU↔CPU round-trip on every decode step. That round-trip
/// blocks the `pending: Option<Array>` async pipeline and is the
/// dominant per-step cost on dense Gemma3 (medgemma planar 42 → 90+ TPS).
/// The on-device `take + dequantize` path keeps everything on `device`,
/// letting MLX fuse the lookup with subsequent layers.
fn gemma3_embedding_lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    let weight_rows = weight.take(ids, 0, device)?;
    let scales_rows = scales.take(ids, 0, device)?;
    let biases_rows = biases.map(|b| b.take(ids, 0, device)).transpose()?;
    let dq = rmlx_mlx::dequantize(
        &weight_rows,
        &scales_rows,
        biases_rows.as_ref(),
        group_size,
        bits,
        mode,
        device,
    )?;
    // Downstream layers (RoPE, attention masks, RmsNormShifted) expect BF16
    // activations. medgemma affine 8b stores scales as BF16 already, so this
    // is a no-op cast in the common case. Keep the guard for robustness.
    if dq.dtype() == Dtype::Bf16 {
        Ok(dq)
    } else {
        dq.astype(Dtype::Bf16, device)
    }
}

// ---------------------------------------------------------------------------
// Local Mlp (mirrors layers::Mlp but uses local Linear with biases)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct Mlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl Mlp {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        // Gemma3 uses gelu_pytorch_tanh (GeGLU variant).
        let gate = self.gate_proj.forward(x, device)?;
        let gate = gelu_tanh(&gate, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}

// ---------------------------------------------------------------------------
// qk_norm_fused — compile_shapeless fusion of (q rms_norm, k rms_norm)
// ---------------------------------------------------------------------------
//
// QK-norm fusion ported from Qwen3 to Gemma3.
//
// Gemma3 uses `RmsNormShifted` (gamma+1 convention; weight pre-shifted at load
// to `weight + 1`). The closure body is identical to Qwen3's qk_norm_fused —
// just `rms_norm(q) + rms_norm(k)`. The shifted weight is passed in at the
// call site (`q_norm.shifted_weight`, `k_norm.shifted_weight`).
//
// Cache: keyed by (in_dtype_tag, device_tag, eps_bits). All Gemma3 layers
// share `rms_norm_eps`, so a single compiled closure handles every layer's
// (B, S, H, D) shape under compile_shapeless. Pattern lifted verbatim from
// `qwen3.rs::qk_norm_fused`.

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
///
/// For Gemma3, callers pass `q_norm.shifted_weight` and `k_norm.shifted_weight`
/// (RmsNormShifted pre-stores `raw_weight + 1.0` at load time, so the closure
/// body is identical to Qwen3's plain-gamma path).
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(super) fn qk_norm_fused(
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
// clip_residual_fused — compile_shapeless port of mlx-lm gemma3_text.clip_residual
// ---------------------------------------------------------------------------
//
// Reference: mlx-lm/mlx_lm/models/gemma3_text.py:125-132
// @partial(mx.compile, shapeless=True)
// def clip_residual(x, y):
// if x.dtype != mx.float16:
// return x + y
// bound = mx.finfo(mx.float16).max
// return mx.clip(x.astype(mx.float32) + y.astype(mx.float32),
// -bound, bound).astype(mx.float16)
//
// medgemma + Gemma3 family run BF16 activations, so the body is `x + y`
// — but compiled. mx.compile traces a single Metal program for the add,
// dropping per-call FFI dispatch (~1-2 µs). 34 layers × 2 residual sites
// × ~85 TPS decode ≈ 68 add calls/step. Per-call savings are small but
// the user directive is explicit: "if copy logic byte-to-byte it should
// perform, at least, the same way".
//
// FP16 path: rMLX has no `clip` op exposed yet (`rmlx_mlx::clip` not in
// public API). For FP16 inputs we fall through to a plain (uncompiled)
// add — math is **NOT** byte-equivalent to mlx-lm in that one case.
// medgemma doesn't hit this. Marked TODO; if a fp16 Gemma3 lands the
// fallthrough must be replaced with a proper clip.

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ClipResidualKey {
    in_dtype_tag: u8,
    device_tag: u8,
}

static CLIP_RESIDUAL_COMPILE_CACHE: OnceLock<
    Mutex<FxHashMap<ClipResidualKey, std::sync::Arc<Closure>>>,
> = OnceLock::new();

fn clip_residual_compile_cache(
) -> &'static Mutex<FxHashMap<ClipResidualKey, std::sync::Arc<Closure>>> {
    CLIP_RESIDUAL_COMPILE_CACHE.get_or_init(|| Mutex::new(FxHashMap::default()))
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn clip_residual_get_or_compile(
    key: ClipResidualKey,
    device: Device,
) -> Result<std::sync::Arc<Closure>> {
    {
        let cache = clip_residual_compile_cache()
            .lock()
            .expect("clip_residual cache poisoned");
        if let Some(cls) = cache.get(&key) {
            return Ok(std::sync::Arc::clone(cls));
        }
    }
    let raw = Closure::from_fn(move |inputs| -> Result<Vec<Array>> {
        if inputs.len() != 2 {
            return Err(Error::Mlx(format!(
                "clip_residual_fused closure: expected 2 inputs (x, y), got {}",
                inputs.len()
            )));
        }
        let mut iter = inputs.into_iter();
        let x = iter.next().expect("x");
        let y = iter.next().expect("y");
        // Non-fp16 path: byte-for-byte port of mlx-lm's `return x + y`.
        let out = add(&x, &y, device)?;
        Ok(vec![out])
    });
    let compiled = compile_shapeless(raw)
        .map_err(|e| Error::Mlx(format!("clip_residual compile_shapeless: {e}")))?;
    let arc = std::sync::Arc::new(compiled);
    let mut cache = clip_residual_compile_cache()
        .lock()
        .expect("clip_residual cache poisoned");
    Ok(std::sync::Arc::clone(
        cache
            .entry(key)
            .or_insert_with(|| std::sync::Arc::clone(&arc)),
    ))
}

/// Compute `x + y` for BF16/F32 inputs via an mx.compile-fused closure.
///
/// Byte-for-byte port of mlx-lm `gemma3_text.clip_residual` for the
/// non-fp16 branch. FP16 inputs currently fall back to a plain add —
/// medgemma doesn't exercise this and rMLX lacks a `clip` wrapper today.
pub(super) fn clip_residual_fused(x: &Array, y: &Array, device: Device) -> Result<Array> {
    if x.dtype() == Dtype::F16 {
        // TODO: port the fp16 branch (cast→add→clip→cast) when a fp16
        // Gemma3 model arrives. mlx-lm's path requires `mx.clip` which is
        // not yet wrapped in rmlx-mlx.
        return add(x, y, device);
    }
    let key = ClipResidualKey {
        in_dtype_tag: qk_norm_dtype_tag(x.dtype()),
        device_tag: qk_norm_device_tag(device),
    };
    let compiled = clip_residual_get_or_compile(key, device)?;
    let mut outs = compiled.apply(&[x, y])?;
    if outs.len() != 1 {
        return Err(Error::Mlx(format!(
            "clip_residual_fused: expected 1 output, got {}",
            outs.len()
        )));
    }
    outs.pop()
        .ok_or_else(|| Error::Mlx("clip_residual_fused: closure returned no outputs".to_owned()))
}
